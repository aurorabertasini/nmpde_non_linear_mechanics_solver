// Define characteristic length
h = 0.05;  // Modify this value as needed

// Define corner points
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

// Define lines with indices
Line(0) = {4, 1};  // Left boundary
Line(1) = {1, 2};  // Bottom boundary
Line(2) = {2, 3};  // Right boundary
Line(3) = {3, 4};  // Top boundary

// Define surface
Line Loop(10) = {0, 1, 2, 3};
Plane Surface(11) = {10};

// Apply physical groups (optional)
Physical Line(0) = {0};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Surface("Domain") = {11};
