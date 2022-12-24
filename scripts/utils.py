import os
import torch
import re
from modules import scripts, script_callbacks, sd_models, shared
from modules.hypernetworks import hypernetwork

class Hypernetwork_Cache:
    def __init__(self, hn):
        self.module_names = list(hn.layers.keys())
        self.layer_structure = hn.layer_structure
        self.activation_func = hn.activation_func
        self.add_layer_norm = hn.add_layer_norm
        self.use_dropout = hn.use_dropout
        self.last_layer_dropout = hn.last_layer_dropout
        self.activate_output = hn.activate_output
        self.sd_checkpoint_name = hn.sd_checkpoint_name
        self.step = hn.step+1
        self.weight_init = hn.weight_init

        def sequential_to_html(seq):
            layers = re.findall(r': ([^\(]+\([^\)]*\))', str(seq))
            layers = map(lambda x: x.split("(", 1), layers)
            html = "<table>"
            for layer, args in layers:
                html += f'<tr><td>{layer}</td><td>({args}</td><tr/>'
            return html+"</table>"
        self.module_htmls = { n : sequential_to_html(hn.layers[n][0].linear) for n in self.module_names}

    def get_module_html(self, module):
        m = self.module_htmls.get(module)
        return m if m else ""


hypernetwork_cache = {}

def get_hypernetwork_names():
    return [x for x in hypernetwork.list_hypernetworks(shared.cmd_opts.hypernetwork_dir).keys()]

def load_hn(hn_name):
    path = shared.hypernetworks.get(hn_name, None)
    hn = hypernetwork.Hypernetwork()
    hn.load(path)
    return hn

def get_module_html_from_cache(hn_name, module_to_display):
    hnc = hypernetwork_cache.get(hn_name)
    if not hnc:
        return ""
    return hnc.get_module_html(int(module_to_display))

def print_hn_info(hn_name, module_to_display):
    hn = load_hn(hn_name)
    hypernetwork_cache[hn_name] = Hypernetwork_Cache(hn)
    return [f'<table>\
    <tr><td>Modules</td><td>{list(hn.layers.keys())}</td></tr>\
    <tr><td>Layer Structure</td><td>{", ".join([str(i) for i in hn.layer_structure])}</td></tr>\
    <tr><td>Activation Func</td><td>{hn.activation_func}</td></tr>\
    <tr><td>Use Layer Normalization</td><td>{hn.add_layer_norm}</td></tr>\
    <tr><td>Use Dropout</td><td>{hn.use_dropout}</td></tr>\
    <tr><td>Last Layer Dropout</td><td>{hn.last_layer_dropout}</td></tr>\
    <tr><td>Activate Output</td><td>{hn.activate_output}</td></tr>\
    <tr><td>&nbsp;</td><td>&nbsp;</td></tr>\
    <tr><td>Checkpoint Name</td><td>{hn.sd_checkpoint_name}</td></tr>\
    <tr><td>Step</td><td>{hn.step+1}</td></tr>\
    <tr><td>Weight Initialization</td><td>{hn.weight_init}</td></tr>\
    <tr><td>&nbsp;</td><td>&nbsp;</td></tr>\
    </table>', hypernetwork_cache[hn_name].get_module_html(int(module_to_display))]

def merge_hn(hna_name, hnb_name, checked_modules, weight, output_name):
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

    if hna.use_dropout != hnb.use_dropout:
        return 'HNs with different dropout setting cannot be merged.'

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