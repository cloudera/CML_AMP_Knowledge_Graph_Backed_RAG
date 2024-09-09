import subprocess


print(subprocess.run(["sh 0_session-install-dependencies/setup.sh"], shell=True))
print(
    subprocess.run(["pip install numpy==1.25.0"], shell=True)
)  # downgrade numpy to 1.25.0 as a safeguard again.
