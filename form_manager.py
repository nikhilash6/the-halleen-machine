# form_manager.py
import json
import gradio as gr
from typing import Any, Callable, Dict, List, Tuple


class ProjectFormRegistry:
    """
    Manages a registry of UI components mapped to paths in the project JSON.
    Allows for:
      1. Centralized definition of UI elements (build time).
      2. Automated loading of values from a JSON string (runtime).
      3. Automated updating of a JSON string from UI values (runtime).
    
    PHASE 2B: Modified to work with dict-only preview_code.
    """
    # def __init__(self):
    #     # Maps path_string -> { 'component': gr.Component, 'default': Any, 'to_ui': func, 'to_json': func }
    #     self._registry: Dict[str, Dict[str, Any]] = {}
    #     # Keep an ordered list of components to pass to event handlers
    #     self._components: List[gr.components.Component] = []
    #     self._paths: List[str] = []
    def __init__(self):
        # List of registry entries to allow multiple components per path
        self._registry: List[Dict[str, Any]] = []

    def add(self, 
            json_path: str, 
            component: gr.components.Component, 
            default: Any = None, 
            to_ui: Callable[[Any], Any] = None, 
            to_json: Callable[[Any], Any] = None,
            is_input: bool = True):
        
        entry = {
            "path": json_path,
            "component": component,
            "default": default,
            "to_ui": to_ui,
            "to_json": to_json,
            "is_input": is_input
        }
        self._registry.append(entry)
        return component

    # def add(self, 
    #         json_path: str, 
    #         component: gr.components.Component, 
    #         default: Any = None, 
    #         to_ui: Callable[[Any], Any] = None, 
    #         to_json: Callable[[Any], Any] = None,
    #         is_input: bool = True):
    #     """
    #     Registers a component.
        
    #     Args:
    #         json_path: Dot-notation path (e.g., "project.width").
    #         component: The Gradio component instance.
    #         default: Default value if path missing in JSON.
    #         to_ui: Optional transform function (JSON -> UI).
    #         to_json: Optional transform function (UI -> JSON).
    #         is_input: If True, this component's value is used to update the JSON. 
    #                   If False (e.g. Markdown), it only displays data.
    #     """
    #     self._registry[json_path] = {
    #         "component": component,
    #         "default": default,
    #         "to_ui": to_ui,
    #         "to_json": to_json,
    #         "is_input": is_input
    #     }
    #     self._components.append(component)
    #     self._paths.append(json_path)
    #     return component

    def get_inputs(self):
        """Returns the list of components registered as inputs (for change events)."""
        return [entry["component"] for entry in self._registry if entry["is_input"]]

    def get_outputs(self):
        """Returns the full list of registered components (for load updates)."""
        return [entry["component"] for entry in self._registry]
    
    # def load_from_json(self, json_data: str | dict):
    #     """
    #     Parses JSON and returns a list of values for all registered components.
    #     Use this as the fn for a load event.
        
    #     PHASE 2B NOTE: Accepts str|dict during migration. Will be dict-only in Phase 2D.
    #     """
    #     # PHASE 2B: Still using _loads() for defensive migration period

    #     data = json_data
        
    #     outputs = []
    #     for path in self._paths:
    #         entry = self._registry[path]
    #         raw_val = self._get_value_by_path(data, path, entry["default"])
            
    #         # Apply transformation if provided
    #         if entry["to_ui"]:
    #             try:
    #                 val = entry["to_ui"](raw_val)
    #             except Exception as e:
    #                 print(f"Error transforming {path} to UI: {e}")
    #                 val = raw_val
    #         else:
    #             val = raw_val
                
    #         outputs.append(val)
            
    #     return outputs

    def load_from_json(self, json_data: str | dict):
        data = json_data
        outputs = []
        
        for entry in self._registry:
            path = entry["path"]
            raw_val = self._get_value_by_path(data, path, entry["default"])
            
            if entry["to_ui"]:
                try:
                    val = entry["to_ui"](raw_val)
                except Exception as e:
                    print(f"Error transforming {path} to UI: {e}")
                    val = raw_val
            else:
                val = raw_val
            
            outputs.append(val)
            
        return outputs

    def update_json(self, current_json: str | dict, *component_values):
        import copy
        data = copy.deepcopy(current_json)
        
        # Get only the entries that correspond to inputs (matching get_inputs order)
        input_entries = [entry for entry in self._registry if entry["is_input"]]
        
        if len(component_values) != len(input_entries):
            print(f"Warning: Form registry mismatch. Got {len(component_values)} values for {len(input_entries)} inputs.")
            return data

        for entry, val in zip(input_entries, component_values):
            path = entry["path"]
            
            if entry["to_json"]:
                try:
                    final_val = entry["to_json"](val)
                except Exception as e:
                    final_val = val
            else:
                final_val = val
                
            self._set_value_by_path(data, path, final_val)
        
        return data
    
    # def update_json(self, current_json: str | dict, *component_values):
    #     """
    #     Updates the provided JSON dict with new values from the UI components.
        
    #     PHASE 2B: NOW RETURNS DICT (not string)
    #     This is the critical fix - preview_code must always be dict.
    #     """
    #     # PHASE 2B: Still using _loads() for defensive migration period
    #     import copy
    #     data = copy.deepcopy(current_json)
        
    #     # Map values back to paths
    #     # Note: component_values corresponds to get_inputs(), so we must match filters
    #     input_paths = [p for p in self._paths if self._registry[p]["is_input"]]
    #     # Build list of unique paths that are inputs (deduplicate self._paths)
    #     seen = set()
    #     input_paths = []
    #     for p in self._paths:
    #         if self._registry[p]["is_input"] and p not in seen:
    #             input_paths.append(p)
    #             seen.add(p)
        

    #     if len(component_values) != len(input_paths):
    #         print(f"Warning: Form registry mismatch. Got {len(component_values)} values for {len(input_paths)} inputs.")
    #         print(f"  Returning unchanged data!")
    #         return data

    #     for i, (path, val) in enumerate(zip(input_paths, component_values)):
    #         entry = self._registry[path]
            

            
    #         # Apply transformation if provided
    #         if entry["to_json"]:
    #             try:
    #                 final_val = entry["to_json"](val)
    #             except Exception as e:
    #                 # print(f"Error transforming {path} to JSON: {e}")
    #                 final_val = val
    #         else:
    #             final_val = val
                
    #         self._set_value_by_path(data, path, final_val)
        
    #     # PHASE 2B FIX: Return dict directly (not _dumps string)
    #     return data




    def _get_value_by_path(self, data: dict, path: str, default: Any):
        keys = path.split('.')
        curr = data
        try:
            for k in keys:
                if isinstance(curr, dict):
                    curr = curr.get(k, None)
                elif isinstance(curr, list):
                    # Attempt to handle list index if path uses numbers
                    idx = int(k)
                    if 0 <= idx < len(curr):
                        curr = curr[idx]
                    else:
                        return default
                else:
                    return default
                
                if curr is None:
                    return default
            return curr
        except Exception:
            return default

    def _set_value_by_path(self, data: dict, path: str, value: Any):
        keys = path.split('.')
        curr = data
        for i, k in enumerate(keys[:-1]):
            # Auto-vivify dictionaries
            if k not in curr or not isinstance(curr[k], (dict, list)):
                # Look ahead to decide if list or dict
                # (Simple assumption: if next key is digit, make list? 
                #  Actually, usually safe to default to dict unless we know schema.
                #  For this project, we are mostly patching into existing structures.)
                curr[k] = {}
            
            curr = curr[k]
            
        last_key = keys[-1]
        if isinstance(curr, dict):
            curr[last_key] = value
        elif isinstance(curr, list):
            try:
                idx = int(last_key)
                if 0 <= idx < len(curr):
                    curr[idx] = value
            except:
                pass
