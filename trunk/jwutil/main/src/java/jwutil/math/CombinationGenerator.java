// CombinationGenerator.java, created Aug 30, 2004 7:24:59 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.math;

import java.math.BigInteger;

/**
 * CombinationGenerator
 * 
 * @author jwhaley
 * @version $Id$
 */
public class CombinationGenerator {
    private int[] a;
    private int n;
    private int r;
    private BigInteger numLeft;
    private BigInteger total;

    /**
     * 
     * @param n
     * @param r
     */
    public CombinationGenerator(int n, int r) {
        if (r > n) {
            throw new IllegalArgumentException(r+" > "+n);
        }
        if (n < 1) {
            throw new IllegalArgumentException(n+" < 1");
        }
        this.n = n;
        this.r = r;
        a = new int[r];
        BigInteger nFact = getFactorial(n);
        BigInteger rFact = getFactorial(r);
        BigInteger nminusrFact = getFactorial(n - r);
        total = nFact.divide(rFact.multiply(nminusrFact));
        reset();
    }

    /**
     * 
     */
    public void reset() {
        for (int i = 0; i < a.length; i++) {
            a[i] = i;
        }
        numLeft = total;
    }

    /**
     * Return number of combinations not yet generated.
     * 
     * @return number of combinations not yet generated
     */
    public BigInteger getNumLeft() {
        return numLeft;
    }

    /**
     * Are there more combinations?
     * 
     * @return if there are more combinations
     */
    public boolean hasMore() {
        return numLeft.compareTo(BigInteger.ZERO) == 1;
    }

    /**
     * Return total number of combinations.
     * 
     * @return total number of combinations
     */
    public BigInteger getTotal() {
        return total;
    }

    /**
     * Compute factorial.
     * 
     * @param n  input number
     * @return  factorial
     */
    private static BigInteger getFactorial(int n) {
        BigInteger fact = BigInteger.ONE;
        for (int i = n; i > 1; i--) {
            fact = fact.multiply(BigInteger.valueOf(i));
        }
        return fact;
    }

    /**
     * Returns the current combination.
     * 
     * @return  current combination
     */
    public int[] getCurrent() {
        return a;
    }
    
    /**
     * Generate next combination (algorithm from Rosen p. 286).
     * 
     * @return  next combination
     */
    public int[] getNext() {
        if (numLeft.equals(total)) {
            numLeft = numLeft.subtract(BigInteger.ONE);
            return a;
        }
        int i = r - 1;
        while (a[i] == n - r + i) {
            i--;
        }
        a[i] = a[i] + 1;
        for (int j = i + 1; j < r; j++) {
            a[j] = a[i] + j - i;
        }
        numLeft = numLeft.subtract(BigInteger.ONE);
        return a;
    }
}
