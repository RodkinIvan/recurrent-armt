"""Recurrent ARMT: associative recurrent memory on top of recurrent LMs (RWKV-7, Mamba, Mamba2, fla models)."""
from transformers import AutoConfig, AutoModelForCausalLM

from .backbones import Backbone, load_base_model, make_backbone
from .gdn2 import GDN2Config, GDN2ForCausalLM
from .memory import AssociativeMemory
from .modeling import ARMTOutput, ARMTState, RecurrentARMTConfig, RecurrentARMTForCausalLM

AutoConfig.register(RecurrentARMTConfig.model_type, RecurrentARMTConfig, exist_ok=True)
AutoModelForCausalLM.register(RecurrentARMTConfig, RecurrentARMTForCausalLM, exist_ok=True)
AutoConfig.register(GDN2Config.model_type, GDN2Config, exist_ok=True)
AutoModelForCausalLM.register(GDN2Config, GDN2ForCausalLM, exist_ok=True)

__all__ = ["ARMTOutput", "ARMTState", "AssociativeMemory", "Backbone", "GDN2Config", "GDN2ForCausalLM", "RecurrentARMTConfig",
           "RecurrentARMTForCausalLM", "load_base_model", "make_backbone"]
