# GOPH 547 Lab #01 Gravity Forward Modelling

Course: GOPH 547 - Gravity and Magnetics
University of Calgary
InstructorL B. Karchewski

Lab Description:
This lab has forward modelling of gravitational potential and the vertical gravity effect for subsurface mass anomalies. A known subsurface mass distribution was given, forward modelling calculates what a gravity survey would measure at the surface.

This lab is split into 2 main parts (Part A, and Part B)

Part A computes and visualizes the gravitational potential and vertical gravity effect for a single buried point mass at multiple survey elevations and grid spacing.

Part B has multiple masses and distributed anomalies. The first part of part B generate sets of five point masses with the same total mass and centre of mass as part A. The second part loads a real density distribution from anomaly_data.py.


Requirements:
- Python 3.8
- numpy
- matplotlib
- scipy

1.) Downloading the Repository Clone this to your local machine by using:

--> git clone https://github.com/MudkipTheGr8/goph547-w2026-lab01-stMD <--

--> cd goph547-w2026-lab00-stMD <--

2.) Setting Up the Virtual Environment/ Installing the Package A Virtual Environment in Windows:

--> python -m venv venv <--

--> venv\Scripts\activate <--

Installing Packages:

--> pip install numpy matplotlib scipy <--

Installing the goph547lab01 package: (From the repository root directory):

--> pip install -e. <--