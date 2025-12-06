"""
Helper module for validating API keys and checking if they are placeholders
"""

def is_valid_api_key(key: str) -> bool:
    """
    Check if an API key is valid (not None, not empty, not a placeholder)
    
    Args:
        key: The API key to validate
        
    Returns:
        bool: True if the key appears to be valid, False otherwise
    """
    if not key:
        return False
    
    key_upper = key.upper()
    
    # Check for common placeholder patterns
    placeholder_patterns = [
        "YOUR_",
        "PLACEHOLDER",
        "REPLACE",
        "INSERT",
        "ADD_YOUR",
        "ENTER_YOUR",
        "PUT_YOUR",
        "_HERE",
        "XXXX",
        "****"
    ]
    
    for pattern in placeholder_patterns:
        if pattern in key_upper:
            return False
    
    # Check if key is too short (most API keys are at least 20 characters)
    if len(key.strip()) < 10:
        return False
    
    return True


def should_suppress_error(status_code: int) -> bool:
    """
    Determine if an HTTP error should be suppressed (not shown to user)
    
    Args:
        status_code: HTTP status code
        
    Returns:
        bool: True if error should be suppressed, False otherwise
    """
    # Suppress authentication/authorization errors (expected when API key is invalid/missing)
    return status_code in [400, 401, 403, 404]
