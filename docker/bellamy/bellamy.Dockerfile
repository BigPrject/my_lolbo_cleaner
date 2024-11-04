# Use an official Conda-based image as the base image
FROM continuumio/miniconda3

# Set up working directory
WORKDIR /workspace

# Copy the environment.yml file into the container
COPY environment.yml .

# Create the Conda environment from the YAML file
RUN conda env create -f environment.yml

# Activate the environment by default for all subsequent RUN commands
RUN echo "source activate <../bellamyenv.yml>" > ~/.bashrc
ENV PATH /opt/conda/envs/<your-env-name>/bin:$PATH

# Copy your project files into the container (modify the source path as necessary)
COPY . /workspace

# Set the default command to run a script or start a shell
CMD ["python", "your_script.py"]