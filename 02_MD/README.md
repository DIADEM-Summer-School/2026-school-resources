# LAMMPS files | DIADEM

## Download LAMMPS-GUI

LAMMPS-GUI can be downloaded from
[https://github.com/akohlmey/lammps-gui/releases/tag/v3.0.7](https://github.com/akohlmey/lammps-gui/releases/tag/v3.0.7).

Alternatively, LAMMPS-GUI can also be downloaded from these links:

- [LAMMPS-GUI (.exe)](https://github.com/akohlmey/lammps-gui/releases/download/v3.0.7/LAMMPS-GUI-Win10-x86_64-v3.0.7.exe)
  for Windows
- [LAMMPS-GUI (.dmg)](https://github.com/akohlmey/lammps-gui/releases/download/v3.0.7/LAMMPS-GUI-macOS-multiarch-v3.0.7.dmg)
  for macOS
- [LAMMPS-GUI (.tar.gz)](https://github.com/akohlmey/lammps-gui/releases/download/v3.0.7/LAMMPS-GUI-Linux-x86_64-v3.0.7.tar.gz)
  for Linux (tarball, preferred option)
- [LAMMPS-GUI (.flatpak)](https://github.com/akohlmey/lammps-gui/releases/download/v3.0.7/LAMMPS-GUI-Linux-x86_64-v3.0.7.flatpak)
  for Linux ([flatpak](https://flatpak.org/))

## Problem with opening LAMMPS-GUI ?

See [this page](https://lammps-gui.lammps.org/installation.html)

## First input

The first input can be downloaded from [argon-lj/NVE/](argon-lj/NVE/input.lmp): 

```bash
# Simple NVE argon simulation

units           lj # Use Lennard-Jones reduced units
dimension       3 # Perform the simulation in 3 spatial dimensions
atom_style      atomic # Atoms are treated as point particles without bonds or molecular topology
boundary        p p p # Use periodic boundary conditions

region          simbox block -20 20 -20 20 -20 20 # Define the simulation box from -20 to 20 in x, y, and z
create_box      1 simbox # Create a simulation box containing one atom type
create_atoms    1 random 600 34134 simbox overlap 0.7 # Randomly create N atoms inside the simulation box

mass            1    1.0 # Assign a mass to atom type 1 (mass = 1.0)
pair_style      lj/cut 4.0 # Use a Lennard-Jones pair potential with a cutoff distance of 4.0
pair_coeff      1    1    1.0 1.0 # Set Lennard-Jones parameters for atom type 1 (epsilon = 1.0 sigma = 1.0)

fix             mynve    all      nve # Integrate the equations of motion
timestep        0.005 # Set the integration timestep to 0.005 in LJ reduced time units

thermo          10 # Print thermodynamic information in log
thermo_style    custom step temp etotal ke pe # Choose what information is printed

# Uncomment to print atom coordinate to file
# dump            mydmp    all      custom 500 dump.lammpstrj id type x y z # Write atom coordinates to file

# Uncomment to generate images
# dump            viz      all      image 500 myimage-*.ppm type type size 800 800 zoom 1.452 shiny 0.5 fsaa yes view 0 0 box yes 0.005 axes no 0.0 0.0 center s 0.483725 0.510373 0.510373
# dump_modify     viz pad 9 boxcolor white backcolor black adiam 1 2 acolor 1 cyan

run             25000
```
