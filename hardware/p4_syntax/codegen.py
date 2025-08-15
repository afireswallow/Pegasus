import sys
import pathlib
import jinja2
from pegasus_parser import parse

def gen_p4(src_text: str, out: pathlib.Path):
    ir = parse(src_text) 
    
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates"),
        trim_blocks=True,
        lstrip_blocks=True
    )
    env.globals['min'] = min
    env.globals['max'] = max

    if ir.model_type == "cnn":
        tpl_name = "cnn_basic.p4.j2"
        print(f"🧬 Detected CNN model type. Using template: {tpl_name}")
    elif ir.model_type == "mlp":
        tpl_name = "mlp_basic.p4.j2"
        print(f"🧬 Detected MLP model type. Using template: {tpl_name}")
    else:
        raise ValueError(f"Unknown model type: {ir.model_type}")

    tpl = env.get_template(tpl_name)
    
    render_context = {"ir": ir}

    out.write_text(tpl.render(render_context), encoding="utf-8")
    print(f"Generated: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python codegen.py <input_config_file> <output_p4_file>")
        sys.exit(1)
    
    config_file, dst_file = sys.argv[1:3]
    src = pathlib.Path(config_file).read_text(encoding="utf-8")
    gen_p4(src, pathlib.Path(dst_file))