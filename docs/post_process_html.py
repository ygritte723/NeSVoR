import sys
import os
import re
from bs4 import BeautifulSoup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

info_dict: dict = dict()
with open("../nesvor/version.py") as fp:
    exec(fp.read(), info_dict)

print("post processing starts")
BASE = sys.argv[1]
if os.path.exists(BASE):
    for f in os.listdir(BASE):
        if f.endswith(".html"):
            p = os.path.join(BASE, f)
            print(p)
            with open(p, "r") as read_file:
                html_source = read_file.read()

            soup = BeautifulSoup(html_source, "html.parser")
            html_tags = soup.find_all(["kbd"])
            for tag in html_tags:
                l = re.sub("[^0-9a-zA-Z]+", "-", tag.text).split("-")
                l = [it for it in l if it]
                tag.attrs["id"] = "-".join(l)

                a = soup.new_tag(
                    "a",
                    attrs={
                        "class": "headerlink",
                        "href": "#" + tag.attrs["id"],
                        "title": "Permalink to this heading",
                    },
                )
                a.string = "¶"
                tag.parent.string.insert_after(a)

            html_string = str(soup)
            html_string = html_string.replace(
                "-version-placeholder-", info_dict["__version__"]
            )

            with open(p, "w") as save_file:
                save_file.write(html_string)


print("post processing finished")
