import onnx

m = onnx.load("yolact_pred.onnx")

print("=== INPUTS ===")
for i in m.graph.input:
    print(i.name)

print("\n=== OUTPUTS ===")
for o in m.graph.output:
    print(o.name)

print("\n=== ALL NODES & OUTPUT TENSORS ===")
for n in m.graph.node:
    print(f"[{n.op_type}]")
    for out in n.output:
        print("  ", out)
