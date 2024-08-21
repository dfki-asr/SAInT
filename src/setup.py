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
              "numpy==1.26.4",
              "torch==2.2.0",
              "fastai==2.7.16",
              "matplotlib==3.9.2",
              "pandas==2.2.2",
              "seaborn==0.13.2",
              "scikit-learn==1.5.1",
              "typing==3.7.4.3",
              "joblib==1.4.2",
              "SALib==1.5.1",
              "lime==0.2.0.1",
              "xgboost==2.1.1",
              "shap==0.46.0",
              "dash==2.17.1",
              "dash-bootstrap-components==1.6.0",
              "kaleido==0.2.1",
              "ipython==8.26.0"
          ],
          extras_require={},
          include_package_data=True,
          zip_safe=False
          )

run_setup()

