import os

import requests


# Check that the current workspace allows workloads to use GPUs
def check_gpu_enabled():
    APIv1 = os.getenv("CDSW_API_URL")
    PATH = "site/config/"
    API_KEY = os.getenv("CDSW_API_KEY")

    url = "/".join([APIv1, PATH])
    res = requests.get(
        url,
        headers={"Content-Type": "application/json"},
        auth=(API_KEY, ""),
    )
    max_gpu_per_engine = res.json().get("max_gpu_per_engine")

    if max_gpu_per_engine < 1:
        print("GPU's are not enabled for this workspace")
        return False
    print("GPUs are enabled in this workspace.")
    return True


if __name__ == "__main__":
    print("Checking the enablement and availibility of GPU in the workspace")
    check_gpu_enabled()
