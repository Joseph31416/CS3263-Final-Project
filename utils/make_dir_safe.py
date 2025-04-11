import re

def make_directory_safe(name: str) -> str:
    """
    Converts a string into a directory-safe format by replacing or removing unsafe characters.
    """
    # Replace slashes and other common filesystem-unsafe characters with underscores
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
    
    # Optionally, strip leading/trailing whitespace or replace multiple underscores with one
    safe_name = re.sub(r'__+', '_', safe_name).strip('_')
    
    return safe_name

if __name__ == "__main__":
    # Example usage
    unsafe_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    safe_name = make_directory_safe(unsafe_name)
    print(f"Original: {unsafe_name}")
    print(f"Safe: {safe_name}")
