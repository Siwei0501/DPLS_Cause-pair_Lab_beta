-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-

                       SUPPEMENTARY DATA 3
                      CHALEARN CAUSE-EFFECT PAIR CHALLENGE
   
              Isabelle Guyon -- isabelle@clopinet.com -- June 2013
                                  
-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-

DISCLAIMER: ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS" 
ISABELLE GUYON AND/OR OTHER CONTRIBUTORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR ANY PARTICULAR PURPOSE, AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S 
INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER CONTRIBUTORS 
BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, 
MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

CONTENTS:

CEdata_train_pairs.csv: Training data 
CEdata_train_publicinfo.csv: Information about Categorical or Numerical variables
CEdata_train_target.csv: Target values

The SUP3 data were drawn using a large pool of real A->B cause-effect pairs of variables from various sources. The role of A and B was reversed in half of them to create B->A pairs.
A random subset of half of the original pairs was selected to create A|B pairs by randomly permuting independently the values of A and B.
The A-B pairs were obtained from a random selection of half of the original pairs to which an algorithm that preserves the marginal distributions while destroying the causal relationships was applied. Pairs of artificially generated dependent variables that are not in a causal relationship were used. Their values were replaced by the values of the real variables in a way that preserves the rank ordering of the values (i.e. the smallest value in the artificial variable is the smallest in the real variable, the second smallest artificial value is the second smallest real value, etc.).

All variable were postprocessed similarly as SUP1 and SUP2 data:
- A random sub-sample of the num_val values original values is drawn without replacement uniformly on a log2 scale between min_size and max_size, where min_size=500 and max_size=8000.Pairs will less than 500 examples are not subsampled.
- Variables with 2 values are considered binary and mapped to 0/1.
- Numerical variables (discrete or continuous) are standardized (the mean is subtracted and then the result is divided by the standard deviation) and then quantized by multiplying the result by 10000 and rounding to the nearest integer.