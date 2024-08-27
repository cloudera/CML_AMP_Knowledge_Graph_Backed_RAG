import sys

from utils.check_dependency import check_unauthenticated_access_to_app_enabled

if check_unauthenticated_access_to_app_enabled() == False:
    sys.exit(
        "Please enable 'Allow applications to be configured with unauthenticated access' from security tab "
    )
