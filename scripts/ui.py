import gradio as gr
from modules import script_callbacks
from modules.ui import create_refresh_button
from scripts.utils import get_hypernetwork_names, load_hn, get_module_html_from_cache, print_hn_info, merge_hn

def on_ui_tabs():
    modules = ["768", "1024", "320", "640", "1280"]
    with gr.Blocks() as main_block:
        with gr.Row():
            with gr.Column(varint="panel"):
                with gr.Row():
                    hna = gr.Dropdown(label="Hypernetwork A", choices=get_hypernetwork_names())
                    create_refresh_button(hna, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hna_html = gr.HTML()
                hna_ls_module = gr.Radio(label="Module to display", choices=modules, value="768", interactive=True)
                hna_ls_html = gr.HTML()
            with gr.Column():
                with gr.Row():
                    hnb = gr.Dropdown(label="Hypernetwork B", choices=get_hypernetwork_names())
                    create_refresh_button(hnb, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hnb_html = gr.HTML()
                hnb_ls_module = gr.Radio(label="Module to display", choices=modules, value="768", interactive=True)
                hnb_ls_html = gr.HTML()
            with gr.Column():
                blend_weight = gr.Slider(label="blend weight (A*(1-w) + B*w)", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True)
                with gr.Row():
                    checked_modules = gr.CheckboxGroup(label="Select modules to merge", value=["768", "320", "640", "1280"], choices=modules, interactive=True)
                output_name = gr.Text(label="Output Hypernetwork Name", interactive=True)
                merge_btn = gr.Button("Merge")
                result_html = gr.HTML()

        hna.change(fn=print_hn_info, inputs=[hna, hna_ls_module], outputs=[hna_html, hna_ls_html])
        hna_ls_module.change(fn=get_module_html_from_cache, inputs=[hna, hna_ls_module], outputs=hna_ls_html)
        hnb.change(fn=print_hn_info, inputs=[hnb, hnb_ls_module], outputs=[hnb_html, hnb_ls_html])
        hnb_ls_module.change(fn=get_module_html_from_cache, inputs=[hnb, hnb_ls_module], outputs=hnb_ls_html)

        merge_btn.click(fn=merge_hn, inputs=[hna, hnb, checked_modules, blend_weight, output_name], outputs=result_html)

        return (main_block, "HNM", "hypernetwork_merger"),

script_callbacks.on_ui_tabs(on_ui_tabs)