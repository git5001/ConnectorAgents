import os


def insert_id_in_filename(filename: str, id_: str) -> str:
    """
    Inserts an id before the file extension in a filename, separated by an underscore.

    Parameters:
        filename (str): The full filename, including the path and extension.
        id_ (str): The id to insert. If None, the filename is returned unchanged.

    Returns:
        str: The updated filename.
    """
    if id_ is None:
        return filename

    # Split the filename into the base (without extension) and extension
    base, ext = os.path.splitext(filename)

    # Create the new filename with the id inserted
    new_filename = f"{base}_{id_}{ext}"

    return new_filename
