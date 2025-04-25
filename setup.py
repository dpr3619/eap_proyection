from setuptools import setup, find_packages

setup(
    name="eap_proyection",
    version="0.1.0",
    description="Proyecciones de la población económicamente activa en Colombia (2025–2040)",
    author="Daniel",
    author_email="tucorreo@example.com",  # opcional
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=True,
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "seaborn>=0.11",
        "plotly>=5.0",
        "pyyaml>=6.0",
        "statsmodels>=0.13",
        "prophet>=1.0",  # si decides usarlo
    ],
    python_requires=">=3.8",
)
