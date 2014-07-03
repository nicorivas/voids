#!/usr/bin/env python

import os
import sys
import math
import time
import argparse 			# Argument parser
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from collections import namedtuple
from scipy.spatial import Delaunay

# Compute Apollonius circles
# Taken from http://rosettacode.org/wiki/Problem_of_Apollonius 

Circle = namedtuple('Circle', 'x, y, r')

def solveApollonius(c1, c2, c3, s1, s2, s3):
	x1, y1, r1 = c1
	x2, y2, r2 = c2
	x3, y3, r3 = c3

	v11 = 2*x2 - 2*x1
	v12 = 2*y2 - 2*y1
	v13 = x1*x1 - x2*x2 + y1*y1 - y2*y2 - r1*r1 + r2*r2
	v14 = 2*s2*r2 - 2*s1*r1

	v21 = 2*x3 - 2*x2
	v22 = 2*y3 - 2*y2
	v23 = x2*x2 - x3*x3 + y2*y2 - y3*y3 - r2*r2 + r3*r3
	v24 = 2*s3*r3 - 2*s2*r2

	w12 = v12/v11
	w13 = v13/v11
	w14 = v14/v11

	w22 = v22/v21-w12
	w23 = v23/v21-w13
	w24 = v24/v21-w14

	P = -w23/w22
	Q = w24/w22
	M = -w12*P-w13
	N = w14 - w12*Q

	a = N*N + Q*Q - 1
	b = 2*M*N - 2*N*x1 + 2*P*Q - 2*Q*y1 + 2*s1*r1
	c = x1*x1 + M*M - 2*M*x1 + P*P + y1*y1 - 2*P*y1 - r1*r1

	# Find a root of a quadratic equation. This requires the circle centers not to be e.g. colinear
	D = b*b-4*a*c
	if D > 0:
		rs = (-b-math.sqrt(D))/(2*a)
	else:
		return False

	xs = M+N*rs
	ys = P+Q*rs

	return Circle(xs, ys, rs)

# Argument parsing
parser = argparse.ArgumentParser(description="Computes channels of a sphere collection. See README.")
parser.add_argument('inputFile',type=argparse.FileType('r'))
parser.add_argument('outputFile',type=argparse.FileType('w'))
parser.add_argument('-p','--periodic',action='store_true',help="enables periodic boundary conditions")
args = parser.parse_args(sys.argv[1:])

start = time.clock()

# Properties
_periodic = True
_dim = 3 
_npart = 0
_debug = False

# Output file
out = args.outputFile

# Read number of points, points and radii from file
f = args.inputFile
_npart = int(f.readline())

points = np.zeros((_npart,3))
radii = np.zeros(_npart)

i = 0
for line in f:
	numbers = line.split()
	points[i] = [float(n) for n in numbers[0:3]]
	radii[i] = float(numbers[3])
	i += 1

print("Input file read: "+str(_npart)+" points")

# Duplicate particles in every direction to account for periodic boundaries in delaunay
if _periodic:
	pointsO = np.array(points,copy=True)
	for dx in range(-1,2):
		for dy in range(-1,2):
			for dz in range(-1,2):
				if dx==0 and dy==0 and dz==0: continue
				points = np.append(points,pointsO+np.array([dx,dy,dz]),axis=0)

# Random points, for testing
'''
points = np.random.uniform(0,1,_dim*_npart)
points = np.split(points,_npart)
'''

# Compute Delaunay triangulation
tri = Delaunay(points)
print("Delaunay triangulation computed")

# Determine neighbors of each particle
neighbors = {}
for pi in range(len(tri.vertices)):
	neighbors[pi] = []
for simplex in tri.vertices:
	neighbors[simplex[0]] += [simplex[1],simplex[2]]
	neighbors[simplex[1]] += [simplex[0],simplex[2]]
	neighbors[simplex[2]] += [simplex[0],simplex[1]]
for pi in range(len(tri.vertices)):
	neighbors[pi] = list(set(neighbors[pi]))

# Cycle through every neighbor triangle of every particle
for pi in range(_npart):
	#print(str(pi)+" "+str(tri.points[pi]))

	# Determine every possible triangle from neighbors
	nn = len(neighbors[pi])
	triangles = []
	for i in range(0,nn):
		for j in range(i+1,nn):
			for k in range(j+1,nn):
				triangles += [[neighbors[pi][i],neighbors[pi][j],neighbors[pi][k]]]
	
	# Cycle through triangles
	for ti in triangles:
		coords = [tri.points[i] for i in ti]

		# Project triangle to it's own plane, move to origin, rotate to x-y
		ab = coords[1]-coords[0]
		ac = coords[2]-coords[0]
		e1 = ab / np.linalg.norm(ab)
		e2_v = np.cross(np.cross(ab,ac),ab)
		e2 = e2_v / np.linalg.norm(e2_v)

		pa = np.array([0,0,0])
		pb = np.array([np.dot(ab,e1),np.dot(ab,e2),0])
		pc = np.array([np.dot(ac,e1),np.dot(ac,e2),0])

		# Size of sides, to determine acuteness
		lab = np.linalg.norm(pb-pa)
		lac = np.linalg.norm(pc-pa)
		lbc = np.linalg.norm(pc-pb)

		# Determine angles of triangle (used before to determine acuteness; too slow)
		#aa = math.acos((lab*lab + lac*lac - lbc*lbc)/(2.0*lab*lac))
		#ab = math.acos((lab*lab + lbc*lbc - lac*lac)/(2.0*lab*lbc))
		#ac = math.acos((lac*lac + lbc*lbc - lab*lab)/(2.0*lac*lbc))
		#if _debug: print(str(aa)+"+"+str(ab)+"+"+str(ac)+"="+str(aa+ab+ac))

		# Determine if the triangle is acute
		sds = sorted([lab,lac,lbc])
		acute = False
		if sds[0]*sds[0] + sds[1]*sds[1] < sds[2]*sds[2]:
			acute = True

		# Define circles	
		ca = Circle(pa[0],pa[1],radii[0])
		cb = Circle(pb[0],pb[1],radii[0])
		cc = Circle(pc[0],pc[1],radii[0])

		# Compute apollonius circle
		cp = solveApollonius(ca,cb,cc,-1,-1,-1)
		if cp is not False:
			if _debug: print(cp)	

			# Plot circles
			
			#cplot1 = plt.Circle((ca.x,ca.y),ca.r)
			#cplot2 = plt.Circle((cb.x,cb.y),cb.r)
			#cplot3 = plt.Circle((cc.x,cc.y),cc.r)
			#cplot4 = plt.Circle((cp.x,cp.y),cp.r)
			#fig = plt.gcf()
			#fig.gca().add_artist(cplot1)
			#fig.gca().add_artist(cplot2)
			#fig.gca().add_artist(cplot3)
			#fig.gca().add_artist(cplot4)
			##fig.savefig('plotcircles.png')
			#plt.axis([min([ca.x,cb.x,cc.x]),max([ca.x,cb.x,cc.x]),min([ca.y,cb.y,cc.y]),max([ca.y,cb.y,cc.y])])
			#plt.show()
			
			# Output
			out.write(str(pi)+" "+str(cp.x)+" "+str(cp.y)+" "+str(cp.r)+" "+str(int(acute))+"\n")

print("Finished. Output written to '"+out.name+"'\nTotal time = "+str(time.clock() - start)+" s.")

