checkKoszulTail = (n, d, bettiTable) -> (
      -- checks if there is an (n,d)-Koszul tail in the betti table
      
      -- nonzero entries are correct
      upperLeft := try bettiTable#(0,{0},0) == 1 else false;
      tail := toList apply(1..n, i -> try bettiTable#(i,{i+d},i+d) == binomial(n,i) else false);
      checkNonzero := all(prepend(upperLeft, tail), i -> i == true);

      -- zero everywhere else
      -- if the sum of all entries is 2^d, then we are good;
      -- otherwise, there is a nonzero entry in a bad spot
      checkZeros := (sum(apply(toList(0..n)**toList(0..d), (i,j)-> try bettiTable#(i,{i+j},i+j) else 0)) == 2^n);
      checkNonzero and checkZeros
)
--tested, works, STG 5/16/2023

koszulTails = (bettiTable) -> (
      -- checks if there is any (n,d)-Koszul tail in the table
      for pair in toList(1..pdim(bettiTable))**toList(1..regularity(bettiTable)) list if checkKoszulTail(pair#0, pair#1, bettiTable) then pair else continue
)

---------------------------------------------------
-------            SCRATCH SPACE            -------
---------------------------------------------------
needsPackage "SimplicialComplexes"

-- boundary of the octahedron (compare to Hal's example in Section 5.2)
kk = ZZ/32003;
R = kk[x_1..x_6];
delta = simplicialComplex({x_1*x_2*x_3, x_1*x_3*x_4, x_1*x_4*x_5, x_1*x_5*x_2,
                              x_6*x_2*x_3, x_6*x_3*x_4, x_6*x_4*x_5, x_6*x_5*x_2});
I = ideal delta;

-- some statistics of the simplicial complex and its Stanley-Reisner ring
print fVector delta;
bettiI = betti res I;
print bettiI;
tails = koszulTails bettiI;
if #tails != 0 then print tails else print "No Koszul tails.";

-- search through all simplicial complexes of on a fixed number of vertices
for subset in subsets(4)