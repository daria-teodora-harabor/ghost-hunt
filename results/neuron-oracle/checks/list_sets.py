import json
j = json.load(open("/workspace/neuron/n1/jobs.json"))
for k, m in j["models"].items():
    print("==", k, m["load"])
    for s, e in m["sets"].items():
        print("  %-28r n=%4d answers=%s alert=%s parent_alert=%s" % (s, len(e["prompts"]), e["answers"] is not None, e["alert"] is not None, e["parent_alert"] is not None))
