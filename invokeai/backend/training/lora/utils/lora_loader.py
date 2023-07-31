import typing

import torch


def find_modules(
    module: torch.nn.Module,
    targets: typing.Set[typing.Type[torch.nn.Module]],
    include_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    exclude_descendants_of: typing.Optional[typing.Set[typing.Type[torch.nn.Module]]] = None,
    memo: typing.Optional[typing.Set[torch.nn.Module]] = None,
    prefix: str = "",
    parent: typing.Optional[torch.nn.Module] = None,
) -> typing.Iterator[typing.Tuple[str, torch.nn.Module, torch.nn.Module]]:
    """Find sub-modules of 'module' that satisfy the search criteria.

    Args:
        module (torch.nn.Module): The base module whose sub-modules will be searched.
        targets (typing.Set[typing.Type[torch.nn.Module]]): The set of module types to search for.
        include_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): If set, then only
            descendants of these types will be searched. exclude_descendants_of takes precedence over
            include_descendants_of.
        exclude_descendants_of (typing.Set[typing.Type[torch.nn.Module]], optional): If set, then the
            descendants of these types will be ignored in the search. exclude_descendants_of takes precedence over
            include_descendants_of.
        memo (typing.Set[torch.nn.Module], optional): A memo to store the set of modules already
            visited in the search. memo is typically only set in recursive calls of this function.
        prefix (str, optional): A prefix that will be added to the module name.
        parent (torch.nn.Module, optional): The parent of 'module'. This is used for tracking the parent in recursive
            calls to this function so that it can be returned along with the module.

    Yields:
        typing.Tuple[str, torch.nn.Module, torch.nn.Module]: A tuple (name, parent, module) that match the search
            criteria.
    """

    if memo is None:
        memo = set()

    if module in memo:
        # We've already visited this module in the search.
        return

    memo.add(module)

    # If we have hit an excluded module type, do not search any further.
    # Note that this takes precedence over include_descendants_of.
    if exclude_descendants_of is not None and any([isinstance(module, class_) for class_ in exclude_descendants_of]):
        return

    # If the include_descendants_of requirement is already satisfied, and this module matches a target class, then all
    # of the search criteria are satisfied, so yield it.
    if include_descendants_of is None and any([isinstance(module, class_) for class_ in targets]):
        yield prefix, parent, module

    # The include_descendants_of requirement is NOT YET satisfied. Check if this module satisfies it.
    updated_include_descendants_of = include_descendants_of
    if include_descendants_of is not None and any([isinstance(module, class_) for class_ in include_descendants_of]):
        # Drop the include_descendants_of requirement if this module satisfied it.
        updated_include_descendants_of = None

    # Recursively search the child modules.
    for child_name, child_module in module.named_children():
        submodule_prefix = prefix + ("." if prefix else "") + child_name
        yield from find_modules(
            module=child_module,
            targets=targets,
            include_descendants_of=updated_include_descendants_of,
            exclude_descendants_of=exclude_descendants_of,
            memo=memo,
            prefix=submodule_prefix,
            parent=module,
        )
