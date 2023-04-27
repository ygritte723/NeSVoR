import time
import argparse
import logging
import re
import os
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..image import Stack, Slice
from ..svort.inference import svort_predict
from ..nesvor.train import train
from ..nesvor.sample import sample_volume, sample_slices
from .io import outputs, inputs
from ..utils import makedirs, log_args, log_result
from ..preprocessing.masking import brain_segmentation
from ..preprocessing import bias_field, motion_estimation
from ..preprocessing.iqa import iqa2d, iqa3d


class Command(object):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.timer: List[Tuple[Optional[str], float]] = []

    def check_args(self) -> None:
        pass

    def get_command(self) -> str:
        return "-".join(
            w.lower() for w in re.findall("[A-Z][^A-Z]*", self.__class__.__name__)
        )

    def new_timer(self, name: Optional[str] = None) -> None:
        t = time.time()
        if len(self.timer) > 1 and self.timer[-1][0] is not None:
            # the previous timer ends
            logging.info(
                "%s finished in %.1f s", self.timer[-1][0], t - self.timer[-1][1]
            )
        if name is None:
            if len(self.timer) == 0:  # begining of command
                pass
            else:  # end of command
                logging.info(
                    "Command 'nesvor %s' finished, overall time: %.1f s",
                    self.get_command(),
                    t - self.timer[0][1],
                )
        else:
            logging.info("%s starts ...", name)
        self.timer.append((name, t))

    def makedirs(self) -> None:
        keys = ["output_slices", "simulated_slices"]
        makedirs([getattr(self.args, k, None) for k in keys])

        keys = ["output_model", "output_volume"]
        for k in keys:
            if getattr(self.args, k, None):
                makedirs(os.path.dirname(getattr(self.args, k)))

        keys = ["output_stack_masks", "output_corrected_stacks"]
        for k in keys:
            if getattr(self.args, k, None):
                for f in getattr(self.args, k):
                    makedirs(os.path.dirname(f))

    def main(self) -> None:
        self.check_args()
        log_args(self.args)
        self.makedirs()
        self.new_timer()
        self.exec()
        self.new_timer()

    def exec(self) -> None:
        raise NotImplementedError("The exec method for Command is not implemented.")


class Reconstruct(Command):
    def check_args(self) -> None:
        # input
        assert (
            self.args.input_slices is not None or self.args.input_stacks is not None
        ), "No image data provided! Use --input-slices or --input-stacks to input data."
        if self.args.input_slices is not None:
            # use input slices
            if (
                self.args.stack_masks is not None
                or self.args.input_stacks is not None
                or self.args.thicknesses is not None
            ):
                logging.warning(
                    "Since <input-slices> is provided, <input-stacks>, <stack_masks> and <thicknesses> would be ignored."
                )
                self.args.stack_masks = None
                self.args.input_stacks = None
                self.args.thicknesses = None
        else:
            # use input stacks
            check_len(self.args, "input_stacks", "stack_masks")
            check_len(self.args, "input_stacks", "thicknesses")
        # output
        if self.args.output_volume is None and self.args.output_model is None:
            logging.warning("Both <output-volume> and <output-model> are not provided.")
        if not self.args.inference_batch_size:
            self.args.inference_batch_size = 8 * self.args.batch_size
        if not self.args.n_inference_samples:
            self.args.n_inference_samples = 2 * self.args.n_samples
        # deformable
        if self.args.deformable:
            if not self.args.single_precision:
                logging.warning(
                    "Fitting deformable model with half precision can be unstable! Try single precision instead."
                )
            if "svort" in self.args.registration:
                logging.warning(
                    "SVoRT can only be used for rigid registration in fetal brain MRI."
                )
        # assessment
        check_cutoff(self.args)
        # registration
        svort_v1_warning(self.args)
        # dtype
        self.args.dtype = torch.float32 if self.args.single_precision else torch.float16

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if "input_stacks" in input_dict and input_dict["input_stacks"]:
            if self.args.segmentation:
                self.new_timer("Segmentation")
                input_dict["input_stacks"] = segment(args, input_dict["input_stacks"])
            if self.args.bias_field_correction:
                self.new_timer("Bias Field Correction")
                input_dict["input_stacks"] = correct_bias_field(
                    args, input_dict["input_stacks"]
                )
            if self.args.metric != "none":
                self.new_timer("Assessment")
                input_dict["input_stacks"], assessment_results = assess(
                    args, input_dict["input_stacks"], False
                )
            self.new_timer("Registration")
            slices = register(args, input_dict["input_stacks"])
        elif "input_slices" in input_dict and input_dict["input_slices"]:
            slices = input_dict["input_slices"]
        else:
            raise ValueError("No data found!")
        self.new_timer("Reconsturction")
        model, output_slices, mask = train(slices, args)
        self.new_timer("Results saving")
        if getattr(input_dict, "volume_mask", None):
            mask = input_dict["volume_mask"]
        output_volume = sample_volume(model, mask, args)
        simulated_slices = (
            sample_slices(model, output_slices, mask, args)
            if getattr(args, "simulated_slices", None)
            else None
        )
        outputs(
            {
                "output_volume": output_volume,
                "mask": mask,
                "output_model": model,
                "output_slices": output_slices,
                "simulated_slices": simulated_slices,
            },
            args,
        )


class SampleVolume(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        self.new_timer("Volume sampling")
        v = sample_volume(input_dict["model"], input_dict["mask"], args)
        self.new_timer("Results saving")
        outputs({"output_volume": v}, args)


class SampleSlices(Command):
    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        self.new_timer("Slices sampling")
        simulated_slices = sample_slices(
            input_dict["model"], input_dict["input_slices"], input_dict["mask"], args
        )
        self.new_timer("Results saving")
        outputs({"simulated_slices": simulated_slices}, args)


class Register(Command):
    def check_args(self) -> None:
        check_len(self.args, "input_stacks", "stack_masks")
        check_len(self.args, "input_stacks", "thicknesses")
        svort_v1_warning(self.args)

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Registration")
        slices = register(args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs({"output_slices": slices}, args)


def register(args: argparse.Namespace, data: List[Stack]) -> List[Slice]:
    svort = args.registration == "svort" or args.registration == "svort-stack"
    vvr = args.registration != "none"
    force_vvr = args.registration == "svort-stack"
    force_scanner = args.scanner_space
    slices = svort_predict(
        data, args.device, args.svort_version, svort, vvr, force_vvr, force_scanner
    )
    return slices


class Segment(Command):
    def check_args(self) -> None:
        if len(self.args.output_stack_masks) == 1:
            folder = self.args.output_stack_masks[0]
            if not (folder.endswith(".nii") or folder.endswith(".nii.gz")):
                # it is a folder
                self.args.output_stack_masks = [
                    os.path.join(folder, "mask_" + os.path.basename(p))
                    for p in self.args.input_stacks
                ]
        check_len(self.args, "input_stacks", "output_stack_masks")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Segmentation")
        seg_stacks = segment(args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs(
            {"output_stack_masks": [stack.get_mask_volume() for stack in seg_stacks]},
            args,
        )


def segment(args: argparse.Namespace, data: List[Stack]) -> List[Stack]:
    data = brain_segmentation.segment(
        data,
        args.device,
        args.batch_size_seg,
        not args.no_augmentation_seg,
        args.dilation_radius_seg,
        args.threshold_small_seg,
    )
    return data


class CorrectBiasField(Command):
    def check_args(self) -> None:
        if len(self.args.output_corrected_stacks) == 1:
            folder = self.args.output_corrected_stacks[0]
            if not (folder.endswith(".nii") or folder.endswith(".nii.gz")):
                # it is a folder
                self.args.output_corrected_stacks = [
                    os.path.join(folder, "corrected_" + os.path.basename(p))
                    for p in self.args.input_stacks
                ]
        check_len(self.args, "input_stacks", "output_corrected_stacks")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        if not ("input_stacks" in input_dict and input_dict["input_stacks"]):
            raise ValueError("No data found!")
        self.new_timer("Bias field correction")
        corrected_stacks = correct_bias_field(args, input_dict["input_stacks"])
        self.new_timer("Results saving")
        outputs(
            {
                "output_corrected_stacks": [
                    stack.get_volume() for stack in corrected_stacks
                ]
            },
            args,
        )


def correct_bias_field(args: argparse.Namespace, stacks: List[Stack]) -> List[Stack]:
    n4_params = {}
    for k in vars(args):
        if k.endswith("_n4"):
            n4_params[k] = getattr(args, k)
    return bias_field.n4_bias_field_correction(stacks, n4_params)


class Assess(Command):
    def check_args(self) -> None:
        check_cutoff(self.args)
        if self.args.metric == "none":
            raise ValueError("--metric should not be none is `assess` command.")

    def exec(self) -> None:
        self.new_timer("Data loading")
        input_dict, args = inputs(self.args)
        self.new_timer("Assessment")
        _, results = assess(self.args, input_dict["input_stacks"], True)
        if args.output_json:
            self.new_timer("Results saving")
            args.output_assessment = results
            outputs({}, args)
            log_result("Assessment results saved to %s" % args.output_json)


def assess(
    args: argparse.Namespace, stacks: List[Stack], print_results=False
) -> Tuple[List[Stack], List[Dict[str, Any]]]:
    metric = args.metric
    descending = True
    if metric == "ncc":
        scores = motion_estimation.ncc(stacks)
    elif metric == "matrix-rank":
        scores = motion_estimation.rank(stacks)
        descending = False
    elif metric == "volume":
        scores = [
            int(
                stack.mask.float().sum().item()
                * stack.resolution_x
                * stack.resolution_y
                * stack.gap
            )
            for stack in stacks
        ]
    elif metric == "iqa2d":
        scores = iqa2d(stacks, args.device)
    elif metric == "iqa3d":
        scores = iqa3d(stacks)
    elif metric == "none":
        return stacks, []
    else:
        raise ValueError("unkown metric for stack assessment")

    n_keep = len(stacks)
    if args.filter_method == "top":
        n_keep = min(n_keep, int(args.cutoff))
    elif args.filter_method == "bottom":
        n_keep = max(0, len(stacks) - int(args.cutoff))
    elif args.filter_method == "percentage":
        n_keep = len(stacks) - int(len(stacks) * min(max(0, args.cutoff), 1))
    elif args.filter_method == "threshold":
        if descending:
            n_keep = sum(score >= args.cutoff for score in scores)
        else:
            n_keep = sum(score <= args.cutoff for score in scores)
    elif args.filter_method == "none":
        pass
    else:
        raise ValueError("unknown filter method")

    sorter = np.argsort(-np.array(scores) if descending else scores)
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    results = []
    for i, (score, rank) in enumerate(zip(scores, inv)):
        results.append(
            dict(
                input_stack=i,
                score=score,
                rank=int(rank),
                excluded=bool(rank >= n_keep),
            )
        )

    template = "\n%15s %15s %15s %15s"
    result_log = "stack assessment results (metric = %s):" % metric + template % (
        "stack",
        "score " + "(" + ("\u2191" if descending else "\u2193") + ")",
        "rank",
        "",
    )
    for i, item in enumerate(results):
        result_log += template % (
            item["input_stack"],
            ("%1.4f" if isinstance(item["score"], float) else "%d") % item["score"],
            item["rank"],
            "excluded" if item["excluded"] else "",
        )
    if print_results:
        log_result(result_log)
    else:
        logging.info(result_log)

    filtered_stacks = [stacks[i] for i in sorter[:n_keep]]
    return filtered_stacks, results


"""warnings and checks"""


def svort_v1_warning(args):
    if "svort" in args.registration and args.svort_version == "v1":
        logging.warning(
            "SVoRT v1 model use a different altas space. If you want to register the image to in the CRL fetal brain atlas space, try the v2 model."
        )


def check_len(args, k1, k2):
    if getattr(args, k1, None) and getattr(args, k2, None):
        assert len(getattr(args, k1)) == len(
            getattr(args, k2)
        ), "The length of {k1} and {k2} are different!"


def check_cutoff(args):
    if args.filter_method != "none" and args.cutoff is None:
        raise ValueError("--cutoff for filtering is not provided!")
