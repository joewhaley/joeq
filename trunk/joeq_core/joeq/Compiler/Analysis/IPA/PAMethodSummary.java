package Compil3r.Analysis.IPA;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import org.sf.javabdd.BDD;

import Clazz.jq_Class;
import Clazz.jq_Field;
import Clazz.jq_Method;
import Clazz.jq_MethodVisitor;
import Clazz.jq_Reference;
import Clazz.jq_Type;
import Compil3r.Analysis.FlowInsensitive.MethodSummary;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteObjectNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.ConcreteTypeNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.GlobalNode;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.UnknownTypeNode;
import Compil3r.Quad.CodeCache;
import Compil3r.Quad.LoadedCallGraph;
import Main.HostedVM;
import Util.Assert;

/**
 * @author jwhaley
 * @version $Id$
 */
public class PAMethodSummary extends jq_MethodVisitor.EmptyVisitor {

    boolean TRACE = false;
    boolean TRACE_RELATIONS = false;
    PrintStream out = System.out;
    
    PA pa;
    
    jq_Method m;
    
    BDD vP;
    BDD L;
    BDD S;
    
    public PAMethodSummary(PA pa, jq_Method m) {
        this.pa = pa;
        this.m = m;
        vP = pa.bdd.zero();
        L = pa.bdd.zero();
        S = pa.bdd.zero();
        visitMethod(m);
    }
    
    public void registerRelations(BDD V1V2context, BDD V1H1context) {
        if (TRACE) out.println("Adding "+m+" with context");
        BDD b;
        if (V1H1context != null) {
            b = vP.and(V1H1context);
            if (PA.VerifyAssertions) {
                if (!b.exist(pa.V1cH1cset).equals(vP)) {
                    System.out.println("m = "+m);
                    System.out.println("vP = "+vP.toStringWithDomains(pa.TS));
                    System.out.println("V1H1context = "+V1H1context.toStringWithDomains());
                    System.out.println("b = "+b.toStringWithDomains());
                    Assert.UNREACHABLE();
                }
            }
        } else {
            if (PA.VerifyAssertions) {
                if ((pa.OBJECT_SENSITIVE || pa.CONTEXT_SENSITIVE) && !vP.isZero()) {
                    System.out.println("m = "+m);
                    System.out.println("vP = "+vP.toStringWithDomains(pa.TS));
                    Assert.UNREACHABLE();
                }
            }
            b = vP.id();
        }
        pa.vP.orWith(b);
        if (V1V2context != null) {
            b = L.and(V1V2context);
            if (PA.VerifyAssertions) {
                Assert._assert(b.exist(pa.V1cV2cset).equals(L));
            }
        } else {
            b = L.id();
        }
        pa.L.orWith(b);
        if (V1V2context != null) {
            b = S.and(V1V2context);
            if (PA.VerifyAssertions) {
                Assert._assert(b.exist(pa.V1cV2cset).equals(S));
            }
        } else {
            b = S.id();
        }
        pa.S.orWith(b);
    }
    
    public void free() {
        if (TRACE) out.println("Freeing "+this.m);
        vP.free(); vP = null;
        L.free(); L = null;
        S.free(); S = null;
    }
    
    void addToVP(BDD V_bdd, Node h) {
        int H_i = pa.Hmap.get(h);
        BDD bdd1 = pa.H1.ithVar(H_i);
        bdd1.andWith(V_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToS(BDD V_bdd, jq_Field f, Collection c) {
        int F_i = pa.Fmap.get(f);
        BDD F_bdd = pa.F.ithVar(F_i);
        for (Iterator k = c.iterator(); k.hasNext(); ) {
            Node node2 = (Node) k.next();
            if (pa.FILTER_NULL && pa.isNullConstant(node2))
                continue;

            int V2_i = pa.Vmap.get(node2);
            BDD bdd1 = pa.V2.ithVar(V2_i);
            bdd1.andWith(F_bdd.id());
            bdd1.andWith(V_bdd.id());
            if (TRACE_RELATIONS) out.println("Adding to S: "+bdd1.toStringWithDomains());
            S.orWith(bdd1);
        }
        F_bdd.free();
    }
    
    void addToL(BDD V_bdd, jq_Field f, Collection c) {
        int F_i = pa.Fmap.get(f);
        BDD F_bdd = pa.F.ithVar(F_i);
        for (Iterator k = c.iterator(); k.hasNext(); ) {
            Node node2 = (Node) k.next();
            int V2_i = pa.Vmap.get(node2);
            BDD bdd1 = pa.V2.ithVar(V2_i);
            bdd1.andWith(F_bdd.id());
            bdd1.andWith(V_bdd.id());
            if (TRACE_RELATIONS) out.println("Adding to L: "+bdd1.toStringWithDomains());
            L.orWith(bdd1);
        }
        F_bdd.free();
    }
    
    public void visitMethod(jq_Method m) {
        if (PA.VerifyAssertions)
            Assert._assert(!pa.newMethodSummaries.containsKey(m));
        pa.newMethodSummaries.put(m, this);
        
        if (TRACE) out.println("Visiting method "+m);
        m.getDeclaringClass().prepare();
        
        int M_i = pa.Mmap.get(m);
        BDD M_bdd = pa.M.ithVar(M_i);
        pa.addToVisited(M_bdd);
        
        if (m.getBytecode() == null) {
            // todo: parameters passed into native methods.
            // build up 'Mret'
            jq_Type retType = m.getReturnType();
            if (retType instanceof jq_Reference) {
                Node node = UnknownTypeNode.get((jq_Reference) retType);
                pa.addToMret(M_bdd, node);
                visitNode(node);
            }
            M_bdd.free();
            return;
        }
        
        MethodSummary ms = MethodSummary.getSummary(CodeCache.getCode(m));
        if (TRACE) out.println("Visiting method summary "+ms);
        
        if (m.isSynchronized() && !m.isStatic()) {
            pa.addToSync(ms.getParamNode(0));
        }
        
        pa.addClassInitializer(ms.getMethod().getDeclaringClass());
        
        // build up 'formal'
        int offset;
        Node thisparam;
        if (ms.getMethod().isStatic()) {
            thisparam = GlobalNode.GLOBAL;
            offset = 0;
        } else {
            thisparam = ms.getParamNode(0);
            offset = 1;
        }
        pa.addToFormal(M_bdd, 0, thisparam);
        int nParams = ms.getNumOfParams();
        for (int i = offset; i < nParams; ++i) {
            Node node = ms.getParamNode(i);
            if (node == null) continue;
            pa.addToFormal(M_bdd, i+1-offset, node);
        }
        
        for (Iterator i = ms.getSyncedVars().iterator(); i.hasNext(); ) {
            Node node = (Node) i.next();
            if (TRACE) out.println("Sync on: "+node);
            pa.addToSync(node);
        }
        
        // build up 'mI', 'actual', 'Iret', 'Ithr'
        for (Iterator i = ms.getCalls().iterator(); i.hasNext(); ) {
            ProgramLocation mc = (ProgramLocation) i.next();
            if (TRACE) out.println("Visiting call site "+mc);
            int I_i = pa.Imap.get(LoadedCallGraph.mapCall(mc));
            BDD I_bdd = pa.I.ithVar(I_i);
            jq_Method target = mc.getTargetMethod();
            if (target.isStatic())
                pa.addClassInitializer(target.getDeclaringClass());
            
            Set thisptr;
            if (target.isStatic()) {
                thisptr = Collections.singleton(GlobalNode.GLOBAL);
                offset = 0;
            } else {
                thisptr = ms.getNodesThatCall(mc, 0);
                offset = 1;
            }
            pa.addToActual(I_bdd, 0, thisptr);
            
            if (mc.isSingleTarget()) {
                // statically-bound, single target call
                pa.addSingleTargetCall(thisptr, mc, I_bdd, target);
            } else {
                // virtual call
                pa.addToMI(M_bdd, I_bdd, target);
            }
            
            jq_Type[] params = mc.getParamTypes();
            for (int k = offset; k < params.length; ++k) {
                if (!params[k].isReferenceType()) continue;
                Set s = ms.getNodesThatCall(mc, k);
                pa.addToActual(I_bdd, k+1-offset, s);
            }
            Node node = ms.getRVN(mc);
            if (node != null) {
                pa.addToIret(I_bdd, node);
            }
            node = ms.getTEN(mc);
            if (!pa.IGNORE_EXCEPTIONS && node != null) {
                pa.addToIthr(I_bdd, node);
            }
            I_bdd.free();
        }
        // build up 'mV', 'vP', 'S', 'L', 'Mret', 'Mthr'
        for (Iterator i = ms.nodeIterator(); i.hasNext(); ) {
            Node node = (Node) i.next();
            
            if (pa.FILTER_NULL && pa.isNullConstant(node))
                continue;
            
            int V_i = pa.Vmap.get(node);
            BDD V_bdd = pa.V1.ithVar(V_i);
            pa.addToMV(M_bdd, V_bdd);
            
            if (ms.getReturned().contains(node)) {
                pa.addToMret(M_bdd, V_i);
            }
            
            if (!pa.IGNORE_EXCEPTIONS && ms.getThrown().contains(node)) {
                pa.addToMthr(M_bdd, V_i);
            }
            
            visitNode(node);
        }
    }
    
    public void visitNode(Node node) {
        if (TRACE) out.println("Visiting node "+node);
       
        if (pa.FILTER_NULL && pa.isNullConstant(node))
            return;
        
        int V_i = pa.Vmap.get(node);
        BDD V_bdd = pa.V1.ithVar(V_i);
        
        if (node instanceof ConcreteTypeNode) {
            addToVP(V_bdd, node);
        } else if (node instanceof ConcreteObjectNode ||
                   node instanceof UnknownTypeNode ||
                   node == GlobalNode.GLOBAL) {
            pa.addToVP(V_bdd, node);
        } else if (node instanceof GlobalNode) {
            int V2_i = pa.Vmap.get(GlobalNode.GLOBAL);
            pa.addToA(V_bdd, V2_i);
            pa.addToA(V2_i, V_i);
        }
        
        for (Iterator j = node.getAllEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Collection c;
            if (e.getValue() instanceof Collection)
                c = (Collection) e.getValue();
            else
                c = Collections.singleton(e.getValue());
            addToS(V_bdd, f, c);
        }
        
        for (Iterator j = node.getAccessPathEdges().iterator(); j.hasNext(); ) {
            Map.Entry e = (Map.Entry) j.next();
            jq_Field f = (jq_Field) e.getKey();
            Collection c;
            if (e.getValue() instanceof Collection)
                c = (Collection) e.getValue();
            else
                c = Collections.singleton(e.getValue());
            addToL(V_bdd, f, c);
            if (node instanceof GlobalNode)
                pa.addClassInitializer(f.getDeclaringClass());
        }
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Method "+m+":");
        sb.append("\nL = ");
        sb.append(L.toStringWithDomains(pa.TS));
        sb.append("\nS = ");
        sb.append(S.toStringWithDomains(pa.TS));
        sb.append("\nvP = ");
        sb.append(vP.toStringWithDomains(pa.TS));
        sb.append('\n');
        return sb.toString();
    }
    
    public static void main(String[] args) {
        HostedVM.initialize();
        CodeCache.AlwaysMap = true;
        jq_Class c = (jq_Class) jq_Type.parseType(args[0]);
        c.load();
        String name = null, desc = null;
        if (args.length > 1) {
            name = args[1];
            if (args.length > 2) {
                desc = args[2];
            }
        }
        Collection methods;
        if (name != null) {
            jq_Method m;
            if (desc != null) {
                m = (jq_Method) c.getDeclaredMember(name, desc);
            } else {
                m = c.getDeclaredMethod(name);
            }
            methods = Collections.singleton(m);
        } else {
            methods = new LinkedList();
            methods.addAll(Arrays.asList(c.getDeclaredStaticMethods()));
            methods.addAll(Arrays.asList(c.getDeclaredInstanceMethods()));
        }
        
        PA pa = new PA();
        pa.initializeBDD(null);
        pa.initializeMaps();
        for (Iterator i = methods.iterator(); i.hasNext(); ) {
            jq_Method m = (jq_Method) i.next();
            if (m.getBytecode() == null) continue;
            PAMethodSummary s = new PAMethodSummary(pa, m);
            System.out.println(s);
        }
    }

}
