import sys
import os
import re
from bs4 import BeautifulSoup


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

            with open(p, "w") as save_file:
                save_file.write(str(soup))


print("post processing finished")
