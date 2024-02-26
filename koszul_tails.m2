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

hasKoszulTail = (n, d, bettiTable) -> (
      --checks if there is a Koszul tail, i.e. a sequence of binomial coefficients,
      --in the betti table. Checks {binomial(n,d)} for 1 <= n <= numVars.
      --NOTE: Expects the Koszul trace in the d-th row
      
      -- run through all (i,j) such that 0 <= i <= numVars and 0 <= j <= d+1
      -- use bettiTable$?{value} (remove braces) to check if entry is nonzero (true if so, false otherwise)
       whereKoszulTraces := for k from 0 to n-1 list checkKoszulTail(n-k, d, bettiTable);
       reverse whereKoszulTraces
)
--tested, works, STG 5/16/2023

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
for pair in toList(1..numgens(R))**toList(1..regularity(I)) do (
      n = pair#0; d = pair#1;
      print("Has a (" | toString(n) | "," | toString(d) | ")-Koszul tail: " | toString any(hasKoszulTail(n, d, bettiI), i -> i == true));
)