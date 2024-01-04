import os
import importlib
import inspect
import pkgutil
from collections import Counter

import inspectors
from inspectors.cli import yes_no_prompt

def is_inspector_class(item):
    from inspectors.base_inspector import BaseInspector
    if not inspect.isclass(item):
        return False
    if inspect.isabstract(item):
        return False
    return issubclass(item, BaseInspector)


def fetch_inspectors():
    inspector_classes = set()
    for module in pkgutil.iter_modules(path=inspectors.__path__, prefix=inspectors.__name__ + "."):
        if module.ispkg:
            continue
        inspector_mod = importlib.import_module(module.name)
        current_classes = inspect.getmembers(inspector_mod, predicate=is_inspector_class)
        inspector_classes.update(current_classes)

    inspector_names = [ins_name for ins_name, _ in inspector_classes]
    duplicates = [item for item, count in Counter(inspector_names).items() if count > 1]

    if len(duplicates) > 0:
        raise RuntimeError(f"Duplicated inspector names {duplicates}. Please rename new inspectors classes.")

    inspectors_by_name = {inspector_name: inspector_class
                          for inspector_name, inspector_class in inspector_classes}

    return inspectors_by_name


INSPECTORS_BY_NAME = fetch_inspectors()


def run_inspectors(runner, dataset, interactive, output_model_script, data_count, logger=None, custom_order=None, **kwargs):
    if custom_order is None:
        custom_order = sorted(INSPECTORS_BY_NAME.keys(), key=lambda x: (-INSPECTORS_BY_NAME[x].PRIORITY, x))
    new_commands = []
    for inspector_name in custom_order:
        inspector = INSPECTORS_BY_NAME[inspector_name](runner, dataset, interactive, logger=logger, **kwargs)
        inspector.add_info(data_count=data_count)
        inspector.run()
        new_commands.extend(inspector.get_new_commands())
    save_model_script(runner, new_commands, output_model_script, interactive, logger)


def save_model_script(runner, new_commands, output_model_script, interactive, logger):
    if len(new_commands) == 0:
        return

    if not output_model_script:
        output_model_script = f"diagnostic_{runner.model_name}.alls"
    new_path = output_model_script
    index = 1
    overwrite = False
    if os.path.exists(new_path) and interactive:
        overwrite = yes_no_prompt(f"Would you like to overwrite file {new_path}?")

    if not overwrite:
        while os.path.exists(new_path):
            basename, ext = os.path.splitext(output_model_script)
            new_path = f"{basename}_{index}{ext}"
            index += 1
        if new_path != output_model_script:
            logger.warning(f"{output_model_script} already exists, saving new model script to {new_path}")
    with open(new_path, "w") as fp:
        for command in new_commands:
            fp.write(f"{command}\n")
