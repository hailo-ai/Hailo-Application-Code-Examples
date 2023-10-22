import importlib
import inspect
import pkgutil
from collections import Counter

import inspectors


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


def run_inspectors(runner, dataset, logger=None, custom_order=None, **kwargs):
    if custom_order is None:
        custom_order = sorted(INSPECTORS_BY_NAME.keys(), key=lambda x: (-INSPECTORS_BY_NAME[x].PRIORITY, x))
        breakpoint()
    for inspector in custom_order:
        INSPECTORS_BY_NAME[inspector](runner, dataset, logger=logger, **kwargs).run()
