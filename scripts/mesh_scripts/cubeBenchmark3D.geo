
cl = 0.09;
Point(1) = {-1, -1, -1, cl};
Point(2) = { 1, -1, -1, cl};
Point(3) = { 1,  1, -1, cl};
Point(4) = {-1,  1, -1, cl};
Point(5) = {-1, -1,  1, cl};
Point(6) = { 1, -1,  1, cl};
Point(7) = { 1,  1,  1, cl};
Point(8) = {-1,  1,  1, cl};


Line(1) = {1, 2};  // bordo "basso" y=-1, z=-1
Line(2) = {2, 3};  
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};  // bordo "alto"  y=-1, z=+1
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9)  = {1, 5}; // collegamenti verticali x=-1
Line(10) = {2, 6}; // collegamenti verticali x=+1
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(13) = {1, 2, 3, 4};
Plane Surface(1) = {13};

Line Loop(14) = {5, 6, 7, 8};
Plane Surface(2) = {14};


Line Loop(15) = {-4, 12, 8, -9};
Plane Surface(3) = {15};


Line Loop(16) = {2, 11, -6, -10};
Plane Surface(4) = {16};

Line Loop(17) = {1, 10, -5, -9};
Plane Surface(5) = {17};

Line Loop(18) = {-3, 11, 7, -12};
Plane Surface(6) = {18};


Surface Loop(19) = {1, 2, 3, 4, 5, 6};
Volume(1) = {19};



Physical Surface(0) = {1}; // z = -1
Physical Surface(1) = {2}; // z = +1
Physical Surface(2) = {3}; // x = -1
Physical Surface(3) = {4}; // x = +1
Physical Surface(4) = {5}; // y = -1
Physical Surface(5) = {6}; // y = +1

Physical Volume(1) = {1};
