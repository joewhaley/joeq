// bddtest.java, created Jul 13, 2003 9:28:32 PM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package simple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BuDDyFactory;

/**
 * bddtest
 * 
 * @author John Whaley
 * @version $Id$
 */
public class bddtest {

    public static void main(String[] args) throws IOException {
        BDDFactory bdd = BuDDyFactory.init(1000000, 10000);
        
        BDDDomain[] domains = bdd.extDomain(new int[] { 10, 8 });
        
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        for (;;) {
            System.out.print("Enter low: ");
            int lo = Integer.parseInt(in.readLine());
            System.out.print("Enter high: ");
            int hi = Integer.parseInt(in.readLine());
            for (int i=0; i<domains.length; ++i) {
                System.out.println(domains[i].varRange(lo, hi).toStringWithDomains());
            }
        }
    }
}
