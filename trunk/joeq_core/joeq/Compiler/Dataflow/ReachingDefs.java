// ReachingDefs.java, created Jun 15, 2003 2:10:14 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compil3r.Dataflow;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import joeq.Clazz.jq_Class;
import joeq.Clazz.jq_Method;
import joeq.Clazz.jq_Type;
import joeq.Compil3r.Quad.BasicBlock;
import joeq.Compil3r.Quad.CodeCache;
import joeq.Compil3r.Quad.ControlFlowGraph;
import joeq.Compil3r.Quad.Quad;
import joeq.Compil3r.Quad.RegisterFactory.Register;
import joeq.Main.HostedVM;
import joeq.Util.BitString;
import joeq.Util.Strings;
import joeq.Util.Collections.Pair;
import joeq.Util.Graphs.EdgeGraph;
import joeq.Util.Graphs.Graph;
import joeq.Util.Templates.List;
import joeq.Util.Templates.ListIterator;

/**
 * ReachingDefs
 * 
 * @author John Whaley
 * @version $Id$
 */
public class ReachingDefs extends Problem {

    Quad[] quads;
    Map transferFunctions;
    RDSet emptySet;
    RDTransferFunction emptyTF;

    static final boolean TRACE = false;

    public void initialize(Graph g) {
        ControlFlowGraph cfg = (ControlFlowGraph) ((EdgeGraph) g).getGraph();
        
        if (TRACE) System.out.println(cfg.fullDump());
        
        // size of bit vector is bounded by the max quad id
        int bitVectorSize = cfg.getMaxQuadID()+1;
        
        if (TRACE) System.out.println("Bit vector size: "+bitVectorSize);
        
        Map regToDefs = new HashMap();
        transferFunctions = new HashMap();
        quads = new Quad[bitVectorSize];
        emptySet = new RDSet(bitVectorSize);
        emptyTF = new RDTransferFunction(bitVectorSize);
        
        List.BasicBlock list = cfg.reversePostOrder(cfg.entry());
        for (ListIterator.BasicBlock i = list.basicBlockIterator(); i.hasNext(); ) {
            BasicBlock bb = i.nextBasicBlock();
            BitString gen = new BitString(bitVectorSize);
            for (ListIterator.Quad j = bb.iterator(); j.hasNext(); ) {
                Quad q = j.nextQuad();
                if (!bb.getExceptionHandlers().isEmpty()) {
                    handleEdges(bb, bb.getExceptionHandlerEntries(), gen, null);
                }
                if (q.getDefinedRegisters().isEmpty()) continue;
                int a = q.getID();
                quads[a] = q;
                for (ListIterator.RegisterOperand k = q.getDefinedRegisters().registerOperandIterator(); k.hasNext(); ) {
                    Register r = k.nextRegisterOperand().getRegister();
                    BitString kill = (BitString) regToDefs.get(r);
                    if (kill == null) regToDefs.put(r, kill = new BitString(bitVectorSize));
                    else gen.minus(kill);
                    kill.set(a);
                }
                gen.set(a);
            }
            RDTransferFunction tf = new RDTransferFunction(gen, new BitString(bitVectorSize));
            handleEdges(bb, bb.getSuccessors(), gen, tf);
        }
        for (Iterator i = transferFunctions.values().iterator(); i.hasNext(); ) {
            RDTransferFunction f = (RDTransferFunction) i.next();
            for (BitString.BitStringIterator j = f.gen.iterator(); j.hasNext(); ) {
                int a = j.nextIndex();
                Quad q = quads[a];
                for (ListIterator.RegisterOperand k = q.getDefinedRegisters().registerOperandIterator(); k.hasNext(); ) {
                    Register r = k.nextRegisterOperand().getRegister();
                    f.kill.or((BitString) regToDefs.get(r));
                }
            }
        }
        if (TRACE) {
            for (Iterator i = transferFunctions.entrySet().iterator(); i.hasNext(); ) {
                Map.Entry e = (Map.Entry) i.next();
                System.out.println(e.getKey());
                System.out.println(e.getValue());
            }
        }
    }

    private void handleEdges(BasicBlock bb, List.BasicBlock bbs, BitString gen, RDTransferFunction defaultTF) {
        for (ListIterator.BasicBlock k = bbs.basicBlockIterator(); k.hasNext(); ) {
            BasicBlock bb2 = k.nextBasicBlock();
            Object edge = new Pair(bb, bb2);
            RDTransferFunction tf = (RDTransferFunction) transferFunctions.get(edge);
            if (tf == null) {
                tf = (defaultTF != null)? defaultTF : new RDTransferFunction(gen.size());
                transferFunctions.put(edge, tf);
            }
            tf.gen.or(gen);
        }
    }

    public static class RDTransferFunction implements TransferFunction {

        protected final BitString gen, kill;

        RDTransferFunction(int size) {
            this.gen = new BitString(size);
            this.kill = new BitString(size);
        }
        RDTransferFunction(BitString g, BitString k) {
            this.gen = g; this.kill = k;
        }

        public String toString() {
            return "Gen: "+gen+Strings.lineSep+"Kill: "+kill;
        }

        /* (non-Javadoc)
         * @see Compil3r.Dataflow.TransferFunction#apply(Compil3r.Dataflow.Fact)
         */
        public Fact apply(Fact f) {
            RDSet r = (RDSet) f;
            BitString s = new BitString(r.reachingDefs.size());
            s.or(r.reachingDefs);
            s.minus(kill);
            s.or(gen);
            return new RDSet(s);
        }
        
    }

    public static class RDSet implements Fact {

        protected final BitString reachingDefs;

        protected RDSet(int size) {
            this.reachingDefs = new BitString(size);
        }
        
        protected RDSet(BitString s) {
            this.reachingDefs = s;
        }

        /* (non-Javadoc)
         * @see Compil3r.Dataflow.Fact#merge(Compil3r.Dataflow.Fact)
         */
        public Fact merge(Fact that) {
            RDSet r = (RDSet) that;
            BitString s = new BitString(this.reachingDefs.size());
            s.or(this.reachingDefs);
            boolean b = s.or(r.reachingDefs);
            if (!b) return this;
            else return new RDSet(s);
        }

        /* (non-Javadoc)
         * @see Compil3r.Dataflow.Fact#equals(Compil3r.Dataflow.Fact)
         */
        public boolean equals(Fact that) {
            return this.reachingDefs.equals(((RDSet) that).reachingDefs);
        }
        
        public String toString() {
            return reachingDefs.toString();
        }
        
    }
    
    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Problem#direction()
     */
    public boolean direction() {
        return true;
    }

    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Problem#boundary()
     */
    public Fact boundary() {
        return emptySet;
    }

    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Problem#interior()
     */
    public Fact interior() {
        return emptySet;
    }

    /* (non-Javadoc)
     * @see Compil3r.Dataflow.Problem#getTransferFunction(java.lang.Object)
     */
    public TransferFunction getTransferFunction(Object e) {
        RDTransferFunction tf = (RDTransferFunction) transferFunctions.get(e);
        if (tf == null) tf = emptyTF;
        return tf;
    }

    public static void main(String[] args) {
        HostedVM.initialize();
        HashSet set = new HashSet();
        for (int i=0; i<args.length; ++i) {
            String s = args[i];
            jq_Class c = (jq_Class) jq_Type.parseType(s);
            c.load();
            set.addAll(Arrays.asList(c.getDeclaredStaticMethods()));
            set.addAll(Arrays.asList(c.getDeclaredInstanceMethods()));
        }
        Problem p = new ReachingDefs();
        Solver s1 = new IterativeSolver();
        Solver s2 = new SortedSetSolver(BBComparator.INSTANCE);
        Solver s3 = new PriorityQueueSolver();
        for (Iterator i=set.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            System.out.println("Method "+m);
            ControlFlowGraph cfg = CodeCache.getCode(m);
            System.out.println(cfg.fullDump());
            solve(cfg, s1, p);
            solve(cfg, s2, p);
            solve(cfg, s3, p);
            dumpResults(cfg, s1);
            compareResults(cfg, s1, s2);
            compareResults(cfg, s2, s3);
        }
    }
    
    public static class BBComparator implements Comparator {

        public static final BBComparator INSTANCE = new BBComparator();
        private BBComparator() {}

        /* (non-Javadoc)
         * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
         */
        public int compare(Object o1, Object o2) {
            if (o1 == o2) return 0;
            int r;
            BasicBlock a, b;
            if (o1 instanceof Pair) {
                a = (BasicBlock) ((Pair) o1).left;
                if (o2 instanceof Pair) {
                    BasicBlock a2 = (BasicBlock) ((Pair) o1).right;
                    b = (BasicBlock) ((Pair) o2).left;
                    BasicBlock b2 = (BasicBlock) ((Pair) o2).right;
                    r = compare(a, b);
                    if (r == 0) r = compare(a2, b2);
                    return r;
                } else {
                    b = (BasicBlock) o2;
                }
            } else {
                a = (BasicBlock) o1;
                if (o2 instanceof Pair) {
                    b = (BasicBlock) ((Pair) o2).left;
                } else {
                    b = (BasicBlock) o2;
                }
            }
            r = compare(a, b);
            if (r == 0) r = (o2 instanceof Pair)?1:-1;
            return r;
        }

        public int compare(BasicBlock bb1, BasicBlock bb2) {
            if (bb1 == bb2) return 0;
            else if (bb1.getID() < bb2.getID()) return -1;
            else return 1;
        }
        
    }
    
    private static void solve(ControlFlowGraph cfg, Solver s, Problem p) {
        s.initialize(p, new EdgeGraph(cfg));
        s.solve();
    }

    private static void dumpResults(ControlFlowGraph cfg, Solver s) {
        for (Iterator i=cfg.reversePostOrderIterator(); i.hasNext(); ) {
            BasicBlock bb = (BasicBlock) i.next();
            Fact r = s.getDataflowValue(bb);
            System.out.println(bb+": "+r);
       }
    }
    
    private static void compareResults(ControlFlowGraph cfg, Solver s1, Solver s2) {
        for (Iterator i=cfg.reversePostOrderIterator(); i.hasNext(); ) {
            BasicBlock bb = (BasicBlock) i.next();
            Fact r1 = s1.getDataflowValue(bb);
            Fact r2 = s2.getDataflowValue(bb);
            if (!r1.equals(r2)) {
                System.out.println("MISMATCH");
                System.out.println(s1.getClass()+" says "+r1);
                System.out.println(s2.getClass()+" says "+r2);
            }
        }
    }

}
