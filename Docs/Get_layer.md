To get **all layers** from a DXF file using the `ezdxf` Python package, you can use the following code:

```python
import ezdxf

# Load the DXF file
doc = ezdxf.readfile("your_file.dxf")

# Access all layers
layers = doc.layers

# Iterate and print layer names
for layer in layers:
    print(f"Layer name: {layer.dxf.name}")
```

### Explanation:
- `doc.layers` gives you a `LayerTable` object.
- Each `layer` in `doc.layers` is a `Layer` object.
- `layer.dxf.name` gives the name of the layer.

### Optional: Get layer properties
If you want more information like color, linetype, etc., you can access them via the `dxf` namespace:

```python
for layer in layers:
    print(f"Name: {layer.dxf.name}, Color: {layer.dxf.color}, Linetype: {layer.dxf.linetype}")
```

Let me know if you want to filter visible layers or get entities by layer!
