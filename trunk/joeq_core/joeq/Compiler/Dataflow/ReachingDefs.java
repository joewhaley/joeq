// ReachingDefs.java, created Jun 15, 2003 2:10:14 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Compiler.Dataflow;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import joeq.Compiler.Quad.BasicBlock;
import joeq.Compiler.Quad.CodeCache;
import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.ControlFlowGraphVisitor;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.RegisterFactory.Register;
import joeq.Main.HostedVM;
import joeq.Util.BitString;
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

    public static class RDVisitor implements ControlFlowGraphVisitor {

        public static boolean DUMP = false;
        
        long totalTime;
        
        /* (non-Javadoc)
         * @see joeq.Compiler.Quad.ControlFlowGraphVisitor#visitCFG(joeq.Compiler.Quad.ControlFlowGraph)
         */
        public void visitCFG(ControlFlowGraph cfg) {
            long time = System.currentTimeMillis();
            Problem p = new ReachingDefs();
            Solver s1 = new IterativeSolver();
            solve(cfg, s1, p);
            time = System.currentTimeMillis() - time;
            totalTime += time;
            if (DUMP) 
                Solver.dumpResults(cfg, s1);
        }
        
        public String toString() {
            return "Total time: "+totalTime+" ms";
        }
    }
    
    Quad[] quads;
    Map transferFunctions;
    BitVectorFact emptySet;
    GenKillTransferFunction emptyTF;

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
        emptySet = new UnionBitVectorFact(bitVectorSize);
        emptyTF = new GenKillTransferFunction(bitVectorSize);
        
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
            GenKillTransferFunction tf = new GenKillTransferFunction(gen, new BitString(bitVectorSize));
            handleEdges(bb, bb.getSuccessors(), gen, tf);
        }
        for (Iterator i = transferFunctions.values().iterator(); i.hasNext(); ) {
            GenKillTransferFunction f = (GenKillTransferFunction) i.next();
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

    private void handleEdges(BasicBlock bb, List.BasicBlock bbs, BitString gen, GenKillTransferFunction defaultTF) {
        for (ListIterator.BasicBlock k = bbs.basicBlockIterator(); k.hasNext(); ) {
            BasicBlock bb2 = k.nextBasicBlock();
            Object edge = new Pair(bb, bb2);
            GenKillTransferFunction tf = (GenKillTransferFunction) transferFunctions.get(edge);
            if (tf == null) {
                tf = (defaultTF != null)? defaultTF : new GenKillTransferFunction(gen.size());
                transferFunctions.put(edge, tf);
            }
            tf.gen.or(gen);
        }
    }

    /* (non-Javadoc)
     * @see Compiler.Dataflow.Problem#direction()
     */
    public boolean direction() {
        return true;
    }

    /* (non-Javadoc)
     * @see Compiler.Dataflow.Problem#boundary()
     */
    public Fact boundary() {
        return emptySet;
    }

    /* (non-Javadoc)
     * @see Compiler.Dataflow.Problem#interior()
     */
    public Fact interior() {
        return emptySet;
    }

    /* (non-Javadoc)
     * @see Compiler.Dataflow.Problem#getTransferFunction(java.lang.Object)
     */
    public TransferFunction getTransferFunction(Object e) {
        TransferFunction tf = (TransferFunction) transferFunctions.get(e);
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
            Solver.dumpResults(cfg, s1);
            Solver.compareResults(cfg, s1, s2);
            Solver.compareResults(cfg, s2, s3);
        }
    }
    
    private static void solve(ControlFlowGraph cfg, Solver s, Problem p) {
        s.initialize(p, new EdgeGraph(cfg));
        s.solve();
    }

}
