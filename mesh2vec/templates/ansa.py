# PYTHON script
# type: ignore
import json

from ansa import constants, base


def make_hg(input_path, output_path, partid):
    referenced_nodes = set()
    referenced_elements = []

    base.InputLSDyna(filename=input_path)

    deck = constants.LSDYNA
    parts = base.CollectEntities(deck, None, "ANSAPART")

    for part in parts:
        part_dict = base.GetEntityCardValues(deck, part, ["__id__", "Name", "PID"])
        if partid != "" and partid != part_dict["PID"]:
            continue

        elements = base.CollectEntities(deck, part, "ELEMENT_SHELL", filter_visible=True)
        for i, element in enumerate(elements):
            fields = [
                "__id__",
                "type",
                "EID",
                "PID",
                "N1",
                "N2",
                "N3",
                "N4",
                "__part__",
            ]
            elem_dict = base.GetEntityCardValues(deck, element, fields)

            # quality etc
            elem_dict["warpage"] = base.ElementQuality(element, "WARP")
            elem_dict["aspect"] = base.ElementQuality(element, "ASPECT")
            elem_dict["skew"] = base.ElementQuality(element, "SKEW")
            elem_dict["area"] = base.CalcShellArea(element)
            elem_dict["normal"] = base.GetNormalVectorOfShell(element)

            # part info
            assert elem_dict["__part__"] == part_dict["__id__"]
            assert int(elem_dict["PID"]) == int(part_dict["PID"])
            elem_dict["part_name"] = part_dict["Name"]

            referenced_elements.append(elem_dict)

            nodes = base.CollectEntities(deck, element, "NODE", filter_visible=True)
            for node in nodes:
                referenced_nodes.add(node)

    fields = ["__id__", "X", "Y", "Z"]
    data = {
        "elements": [e for e in referenced_elements],
        "nodes": [base.GetEntityCardValues(deck, n, fields) for n in referenced_nodes],
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(output_path + " written!")
