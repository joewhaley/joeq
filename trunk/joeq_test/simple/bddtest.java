// bddtest.java, created Jul 13, 2003 9:28:32 PM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package simple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;

/**
 * bddtest
 * 
 * @author John Whaley
 * @version $Id$
 */
public class bddtest {

    public static void main(String[] args) throws IOException {
        BDDFactory bdd = BDDFactory.init(1000000, 10000);
        
        BDDDomain[] domains = bdd.extDomain(new int[] { 10, 8 });
        int[] order = new int[bdd.varNum()];
        
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        for (;;) {
            buildRandomPermutation(order);
            printPermutation(order);
            bdd.setVarOrder(order);
            System.out.print("Enter low: ");
            int lo = Integer.parseInt(in.readLine());
            System.out.print("Enter high: ");
            int hi = Integer.parseInt(in.readLine());
            for (int i=0; i<domains.length; ++i) {
                BDD b = domains[i].varRange(lo, hi);
                System.out.println(b.toStringWithDomains()+" = "+b.nodeCount()+" nodes");
                buildRandomPermutation(order);
                printPermutation(order);
                bdd.setVarOrder(order);
                System.out.println(b.toStringWithDomains()+" = "+b.nodeCount()+" nodes");
                buildRandomPermutation(order);
                printPermutation(order);
                bdd.setVarOrder(order);
                System.out.println(b.toStringWithDomains()+" = "+b.nodeCount()+" nodes");
                buildRandomPermutation(order);
                printPermutation(order);
                bdd.setVarOrder(order);
                System.out.println(b.toStringWithDomains()+" = "+b.nodeCount()+" nodes");
            }
        }
    }
    
    static void printPermutation(int[] a) {
        for (int i=0; i<a.length; ++i) {
            System.out.print(a[i]+" ");
        }
        System.out.println();
    }
    
    static void buildRandomPermutation(int[] a) {
        Arrays.fill(a, -1);
        int n = 0;
        java.util.Random r = new java.util.Random();
        while (n < a.length) {
            int k = r.nextInt(a.length);
            if (a[k] == -1) {
                a[k] = n++;
            }
        }
    }
}
