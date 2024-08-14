from setuptools import setup, find_packages

packages = find_packages(include=["src"], exclude=["data", "outputs", "templates"])

def run_setup():
    setup(name="SAInT",
          version="1.0.0",
          description="SAInT: An Interactive Tool for Sensitivity Analysis In The Loop",
          url="https://github.com/dfki-asr/c3di/SAInT",
          author="Manuela Schuler",
          author_email="Manuela.Schuler@dfki.de",
          packages=packages,
          python_requires=">=3.8",
          install_requires=[
              "numpy",
              "torch",
              "fastai",
              "matplotlib",
              "pandas",
              "seaborn",
              "scikit-learn",
              "typing",
              "joblib",
              "SALib",
              "lime",
              "xgboost",
              "shap",
              "dash",
              "dash-bootstrap-components",
              "kaleido"
          ],
          extras_require={},
          include_package_data=True,
          zip_safe=False
          )

run_setup()

