import os
import gradio as gr
import torch
from modules import scripts, script_callbacks, sd_models, shared
from modules.ui import create_refresh_button
from modules.hypernetworks import hypernetwork
from scripts.utils import get_hypernetwork_names, load_hn, layer_structure_to_html, print_hn_info, merge_hn

def on_ui_tabs():
    with gr.Blocks() as main_block:
        with gr.Row():
            with gr.Column(varint="panel"):
                with gr.Row():
                    hna = gr.Dropdown(label="Hypernetwork A", choices=get_hypernetwork_names())
                    create_refresh_button(hna, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hna_html = gr.HTML()
            with gr.Column():
                with gr.Row():
                    hnb = gr.Dropdown(label="Hypernetwork B", choices=get_hypernetwork_names())
                    create_refresh_button(hnb, get_hypernetwork_names, lambda: {"choices": get_hypernetwork_names()}, "refresh_hn_models")
                hnb_html = gr.HTML()
            with gr.Column():
                blend_weight = gr.Slider(label="blend weight (A*(1-w) + B*w)", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True)
                with gr.Row():
                    checked_modules = gr.CheckboxGroup(label="Select modules to merge", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], interactive=True)
                output_name = gr.Text(label="Output Hypernetwork Name", interactive=True)
                merge_btn = gr.Button("Merge")
                result_html = gr.HTML()

        hna.change(fn=print_hn_info, inputs=hna, outputs=hna_html)
        hnb.change(fn=print_hn_info, inputs=hnb, outputs=hnb_html)

        merge_btn.click(fn=merge_hn, inputs=[hna, hnb, checked_modules, blend_weight, output_name], outputs=result_html)

        return (main_block, "HNM", "hypernetwork_merger"),

script_callbacks.on_ui_tabs(on_ui_tabs)