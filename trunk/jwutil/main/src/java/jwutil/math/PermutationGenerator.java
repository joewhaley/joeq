// PermutationGenerator.java, created Mar 19, 2004 10:12:30 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.math;

import java.math.BigInteger;

/**
 * PermutationGenerator
 * 
 * @author jwhaley
 * @version $Id$
 */
public class PermutationGenerator {
    
    private int[] a;
    private BigInteger numLeft;
    private BigInteger total;

    public PermutationGenerator(int n) {
        if (n < 1) {
            throw new IllegalArgumentException();
        }
        a = new int[n];
        total = getFactorial(n);
        reset();
    }

    public void reset() {
        for (int i = 0; i < a.length; i++) {
            a[i] = i;
        }
        numLeft = total;
    }

    /**
     * Return number of permutations not yet generated
     */
    public BigInteger getNumLeft() {
        return numLeft;
    }

    /**
     * Return total number of permutations
     */
    public BigInteger getTotal() {
        return total;
    }

    /**
     * Are there more permutations?
     */
    public boolean hasMore() {
        return numLeft.compareTo(BigInteger.ZERO) == 1;
    }

    /**
     * Compute factorial
     */
    public static BigInteger getFactorial(int n) {
        BigInteger fact = BigInteger.ONE;
        for (int i = n; i > 1; i--) {
            fact = fact.multiply(BigInteger.valueOf(i));
        }
        return fact;
    }

    /**
     * Generate next permutation (algorithm from Rosen p. 284)
     */
    public int[] getNext() {
        if (numLeft.equals(total)) {
            numLeft = numLeft.subtract(BigInteger.ONE);
            return a;
        }
        int temp;
        // Find largest index j with a[j] < a[j+1]
        int j = a.length - 2;
        while (a[j] > a[j + 1]) {
            j--;
        }
        // Find index k such that a[k] is smallest integer
        // greater than a[j] to the right of a[j]
        int k = a.length - 1;
        while (a[j] > a[k]) {
            k--;
        }
        // Interchange a[j] and a[k]
        temp = a[k];
        a[k] = a[j];
        a[j] = temp;
        // Put tail end of permutation after jth position in increasing order
        int r = a.length - 1;
        int s = j + 1;
        while (r > s) {
            temp = a[s];
            a[s] = a[r];
            a[r] = temp;
            r--;
            s++;
        }
        numLeft = numLeft.subtract(BigInteger.ONE);
        return a;
    }
}
