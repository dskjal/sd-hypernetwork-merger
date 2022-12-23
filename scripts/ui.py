import os
import gradio as gr
import torch
from modules import scripts, script_callbacks, sd_models, shared
from modules.ui import create_refresh_button
from modules.hypernetworks import hypernetwork

def get_hypernetwork_names():
    return [x for x in hypernetwork.list_hypernetworks(shared.cmd_opts.hypernetwork_dir).keys()]

def on_ui_tabs():
    with gr.Blocks() as main_block:
        # UI
        with gr.Row():
            with gr.Column(varint="panel"):
                with gr.Row():
                    hna = gr.Dropdown(label="Hypernetwork A", choices=get_hypernetwork_names())
                    create_refresh_button(hna, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hna_html = gr.HTML()
            with gr.Column():
                with gr.Row():
                    hnb = gr.Dropdown(label="Hypernetwork B", choices=[x for x in shared.hypernetworks.keys()])
                    create_refresh_button(hnb, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hnb_html = gr.HTML()
            with gr.Column():
                blend_weight = gr.Slider(label="blend weight (A*(1-w) + B*w)", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True)
                with gr.Row():
                    checked_modules = gr.CheckboxGroup(label="Select modules to merge", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], interactive=True)
                output_name = gr.Text(label="Output Hypernetwork Name", interactive=True)
                merge_btn = gr.Button("Merge")
                result_html = gr.HTML()

        # Logic
        def load_hn(hn_name):
            path = shared.hypernetworks.get(hn_name, None)
            hn = hypernetwork.Hypernetwork()
            hn.load(path)
            return hn
        def print_hn_info(hn_name):
            hn = load_hn(hn_name)
            return f'<table>\
            <tr><td>Modules</td><td>{list(hn.layers.keys())}</td></tr>\
            <tr><td>Layer Structure</td><td>{hn.layer_structure}</td></tr>\
            <tr><td>Activation Func</td><td>{hn.activation_func}</td></tr>\
            <tr><td>Use Layer Normalization</td><td>{hn.add_layer_norm}</td></tr>\
            <tr><td>Use Dropout</td><td>{hn.use_dropout}</td></tr>\
            <tr><td>Last Layer Dropout</td><td>{hn.last_layer_dropout}</td></tr>\
            <tr><td>Activate Output</td><td>{hn.activate_output}</td></tr>\
            </table>'
        hna.change(fn=print_hn_info, inputs=hna, outputs=hna_html)
        hnb.change(fn=print_hn_info, inputs=hnb, outputs=hnb_html)

        def merge(hna_name, hnb_name, checked_modules, weight, output_name):
            if not hna_name or hna_name == '':
                return 'Select Hypernetwork A'
            if not hnb_name or hnb_name == '':
                return 'Select Hypernetwork B'

            hna = load_hn(hna_name)
            if not hna:
                return f'Loading hypernetwork {hna_name} is failed'
            hnb = load_hn(hnb_name)
            if not hnb:
                return f'Loading hypernetwork {hnb_name} is failed'

            if hna.layer_structure != hnb.layer_structure:
                return 'HNs with different layer structures cannot be merged.'

            if hna.add_layer_norm != hnb.add_layer_norm:
                return 'HNs with different layer normalization setting cannot be merged.'

            def merge_module(m1, m2, weight):
                copy_type = [torch.nn.Linear, torch.nn.LayerNorm]
                for i in range(len(m1.linear)):
                    l1 = m1.linear[i]
                    l2 = m2.linear[i]
                    if type(l1) in copy_type and type(l2) in copy_type:
                        with torch.no_grad():
                            m1.linear[i].weight.data.lerp_(l2.weight.data, weight)
                            m1.linear[i].bias.data.lerp_(l2.bias.data, weight) 
                return m1

            msg = ''
            for module in checked_modules:
                module = int(module)
                m1 = hna.layers.get(module)
                if m1 == None:
                    msg += f'{hna_name} has no {module} module. Merging module {module} is skipped.<br>'
                    continue
                m2 = hnb.layers.get(module)
                if m2 == None:
                    msg += f'{hnb_name} has no {module} module. Merging module {module} is skipped.<br>'
                    continue

                hna.layers[module] = [merge_module(m1[0], m2[0], weight), merge_module(m1[1], m2[1], weight)]

            if not output_name or output_name == "":
                output_name = f'mh-{hna_name}-{hnb_name}-{weight}'
            hna.save(os.path.join(shared.cmd_opts.hypernetwork_dir, f'{output_name}.pt'))

            msg += 'Merge succeeded.'
            return msg

        merge_btn.click(fn=merge, inputs=[hna, hnb, checked_modules, blend_weight, output_name], outputs=result_html)

        return (main_block, "HNM", "hypernetwork_merger"),

script_callbacks.on_ui_tabs(on_ui_tabs)