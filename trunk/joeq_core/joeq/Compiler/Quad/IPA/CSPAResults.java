// CSPAResults.java, created Aug 7, 2003 12:34:24 AM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Compil3r.Quad.IPA;

import java.io.*;
import java.util.StringTokenizer;

import org.sf.javabdd.*;

import Compil3r.Quad.MethodSummary;
import Compil3r.Quad.MethodSummary.Node;
import Main.HostedVM;
import Util.Assert;
import Util.Collections.IndexMap;

/**
 * CSPAResults
 * 
 * @author John Whaley
 * @version $Id$
 */
public class CSPAResults {

    BDDFactory bdd;

    IndexMap variableIndexMap;
    IndexMap heapobjIndexMap;

    BDDDomain V1c, V1o, H1c, H1o;
    BDD pointsTo; // V1c x V1o x H1c x H1o

    public static class SetOfContexts {
        
    }

    public void load(String fn) throws IOException {
        DataInput di;
        
        di = new DataInputStream(new FileInputStream(fn+".config"));
        readConfig(di);
        
        this.pointsTo = bdd.load(fn+".bdd");
        
        di = new DataInputStream(new FileInputStream(fn+".vars"));
        variableIndexMap = readIndexMap("Variable", di);
        
        di = new DataInputStream(new FileInputStream(fn+".heap"));
        heapobjIndexMap = readIndexMap("Heap", di);
    }

    void readConfig(DataInput in) throws IOException {
        String s = in.readLine();
        StringTokenizer st = new StringTokenizer(s);
        int VARBITS = Integer.parseInt(st.nextToken());
        int HEAPBITS = Integer.parseInt(st.nextToken());
        int FIELDBITS = Integer.parseInt(st.nextToken());
        int CLASSBITS = Integer.parseInt(st.nextToken());
        int CONTEXTBITS = Integer.parseInt(st.nextToken());
        int[] domainBits;
        int[] domainSpos;
        BDDDomain[] bdd_domains;
        domainBits = new int[] {VARBITS, CONTEXTBITS,
                                VARBITS, CONTEXTBITS,
                                FIELDBITS,
                                HEAPBITS, CONTEXTBITS,
                                HEAPBITS, CONTEXTBITS};
        domainSpos = new int[domainBits.length];
        
        long[] domains = new long[domainBits.length];
        for (int i=0; i<domainBits.length; ++i) {
            domains[i] = (1L << domainBits[i]);
        }
        bdd_domains = bdd.extDomain(domains);
        V1o = bdd_domains[0];
        V1c = bdd_domains[1];
        H1o = bdd_domains[5];
        H1c = bdd_domains[6];
        
        boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true");
        String ordering = System.getProperty("bddordering", "FD_H2cxH2o_V2cxV1cxV2oxV1o_H1cxH1o");
        
        int[] varorder = CSPA.makeVarOrdering(bdd, domainBits, domainSpos,
                                              reverseLocal, ordering);
        bdd.setVarOrder(varorder);
        bdd.enableReorder();
    }
    
    IndexMap readIndexMap(String name, DataInput in) throws IOException {
        int size = Integer.parseInt(in.readLine());
        IndexMap m = new IndexMap(name, size);
        for (int i=0; i<size; ++i) {
            String s = in.readLine();
            StringTokenizer st = new StringTokenizer(s);
            Node n = MethodSummary.readNode(st);
            int j = m.get(n);
            System.out.println(i+" = "+n);
            Assert._assert(i == j);
        }
        return m;
    }

    public static void main(String[] args) throws IOException {
        HostedVM.initialize();
        
        int nodeCount = 1000000;
        int cacheSize = 100000;
        BDDFactory bdd = BDDFactory.init(nodeCount, cacheSize);
        bdd.setMaxIncrease(nodeCount/4);
        CSPAResults r = new CSPAResults(bdd);
        r.load("cspa");
    }

    public CSPAResults(BDDFactory bdd) {
        this.bdd = bdd;
    }

}
