/*
 * Created on Jun 24, 2004
 */
package joeq.Compiler.Analysis.IPA;

import java.io.*;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Collections;
import joeq.Compiler.Analysis.FlowInsensitive.MethodSummary.Node;
import joeq.Main.HostedVM;
import joeq.Util.Assert;
import joeq.Util.Collections.IndexMap;
import joeq.Util.Collections.IndexedMap;
import joeq.Util.IO.SystemProperties;
import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.dom.ASTParser;
import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;


/**
 * Derive input relations directly from source using the Eclipse AST package.
 * 
 * @author jimz
 */
public class PAFromSource {
    PrintStream out = System.out;
    
    boolean TRACE_RELATIONS = !System.getProperty("pas.tracerelations", "no").equals("no");
    
    IndexMap/*Node*/ Vmap;
    IndexMap/*ProgramLocation*/ Imap;
    IndexedMap/*Node*/ Hmap;
    IndexMap/*jq_Field*/ Fmap;
    IndexMap/*jq_Reference*/ Tmap;
    IndexMap/*jq_Method*/ Nmap;
    IndexMap/*jq_Method*/ Mmap;
    //PathNumbering vCnumbering; // for context-sensitive
    //PathNumbering hCnumbering; // for context-sensitive
    //PathNumbering oCnumbering; // for object-sensitive
    
    BDDFactory bdd;
    
    BDDDomain V1, V2, I, H1, Z, F, T1, T2, M; // H2, N, M2
    //BDDDomain V1c[], V2c[], H1c[], H2c[];
    
    int V_BITS=18, I_BITS=16, H_BITS=15, Z_BITS=5, F_BITS=13, T_BITS=12, N_BITS=13, M_BITS=14;
    //int VC_BITS=0, HC_BITS=0;
    //int MAX_VC_BITS = Integer.parseInt(System.getProperty("pas.maxvc", "61"));
    //int MAX_HC_BITS = Integer.parseInt(System.getProperty("pas.maxhc", "0"));

    BDD A;     // V1xV2, arguments and return values   (+context)
    BDD vP;     // V1xH1, variable points-to            (+context)
    BDD S;      // (V1xF)xV2, stores                    (+context)
    BDD L;      // (V1xF)xV2, loads                     (+context)
    BDD vT;     // V1xT1, variable type                 (no context)
    BDD hT;     // H1xT2, heap type                     (no context)
    BDD aT;     // T1xT2, assignable types              (no context)
    //BDD cha;    // T2xNxM, class hierarchy information  (no context)
    BDD actual; // IxZxV2, actual parameters            (no context)
    BDD formal; // MxZxV1, formal parameters            (no context)
    BDD Iret;   // IxV1, invocation return value        (no context)
    BDD Mret;   // MxV2, method return value            (no context)
    BDD Ithr;   // IxV1, invocation thrown value        (no context)
    BDD Mthr;   // MxV2, method thrown value            (no context)
    //BDD mI;     // MxIxN, method invocations            (no context)
    //BDD mV;     // MxV, method variables                (no context)
    //BDD sync;   // V, synced locations                  (no context)

    //BDD fT;     // FxT2, field types                    (no context)
    //BDD fC;     // FxT2, field containing types         (no context)

    //BDD hP;     // H1xFxH2, heap points-to              (+context)
    BDD IE;     // IxM, invocation edges                (no context)
    //BDD IEcs;   // V2cxIxV1cxM, context-sensitive invocation edges
    //BDD vPfilter; // V1xH1, type filter                 (no context)
    //BDD hPfilter; // H1xFxH2, type filter               (no context)
    //BDD NNfilter; // H1, non-null filter                (no context)
    //BDD IEfilter; // V2cxIxV1cxM, context-sensitive edge filter
    
    //BDDPairing V1toV2, V2toV1, H1toH2, H2toH1, V1H1toV2H2, V2H2toV1H1;
    //BDDPairing V1ctoV2c, V1cV2ctoV2cV1c, V1cH1ctoV2cV1c;
    //BDDPairing T2toT1, T1toT2;
    //BDDPairing H1toV1c[], V1ctoH1[]; 
    //BDD V1csets[], V1cH1equals[];
    BDD V1set, V2set, H1set, T1set, T2set, Fset, Mset, Iset, Zset; //H2set, Nset, 
    BDD V1V2set, V1Fset, V2Fset, V1FV2set, V1H1set, H1Fset; //, H2Fset, H1H2set, H1FH2set
    BDD IMset, MZset; //INset, INH1set, INT2set, T2Nset, 
    //BDD V1cset, V2cset, H1cset, H2cset, V1cV2cset, V1cH1cset, H1cH2cset;
    //BDD V1cdomain, V2cdomain, H1cdomain, H2cdomain;

    String varorder = System.getProperty("bddordering");
    //int MAX_PARAMS = Integer.parseInt(System.getProperty("pas.maxparams", "4"));
    int bddnodes = Integer.parseInt(System.getProperty("bddnodes", "2500000"));
    int bddcache = Integer.parseInt(System.getProperty("bddcache", "200000"));
    int bddminfree = Integer.parseInt(System.getProperty("bddminfree", "20"));
    boolean reverseLocal = System.getProperty("bddreverse", "true").equals("true"); 
    
    BDDDomain makeDomain(String name, int bits) {
        Assert._assert(bits < 64);
        BDDDomain d = bdd.extDomain(new long[] { 1L << bits })[0];
        d.setName(name);
        return d;
    }
    
    public void initializeBDD(String bddfactory) {
        //USE_VCONTEXT = VC_BITS > 0;
        //USE_HCONTEXT = HC_BITS > 0;
        
        //if (USE_VCONTEXT || USE_HCONTEXT) 
        //  bddnodes *= 2;

        if (bddfactory == null)
            bdd = BDDFactory.init(bddnodes, bddcache);
        else
            bdd = BDDFactory.init(bddfactory, bddnodes, bddcache);

        bdd.setMaxIncrease(bddnodes/4);
        bdd.setMinFreeNodes(bddminfree);

        V1 = makeDomain("V1", V_BITS);
        V2 = makeDomain("V2", V_BITS);
        I = makeDomain("I", I_BITS);
        H1 = makeDomain("H1", H_BITS);
        //H2 = makeDomain("H2", H_BITS);
        Z = makeDomain("Z", Z_BITS);
        F = makeDomain("F", F_BITS);
        T1 = makeDomain("T1", T_BITS);
        T2 = makeDomain("T2", T_BITS);
        //N = makeDomain("N", N_BITS);
        M = makeDomain("M", M_BITS);
        //M2 = makeDomain("M2", M_BITS);
        
        /*
        if (CONTEXT_SENSITIVE || OBJECT_SENSITIVE || THREAD_SENSITIVE) {
            V1c = new BDDDomain[1];
            V2c = new BDDDomain[1];
            V1c[0] = makeDomain("V1c", VC_BITS);
            V2c[0] = makeDomain("V2c", VC_BITS);
        } else if (CARTESIAN_PRODUCT && false) {
            V1c = new BDDDomain[MAX_PARAMS];
            V2c = new BDDDomain[MAX_PARAMS];
            for (int i = 0; i < V1c.length; ++i) {
                V1c[i] = makeDomain("V1c"+i, H_BITS + HC_BITS);
            }
            for (int i = 0; i < V2c.length; ++i) {
                V2c[i] = makeDomain("V2c"+i, H_BITS + HC_BITS);
            }
        } else {
            V1c = V2c = new BDDDomain[0];
        }
        if (USE_HCONTEXT) {
            H1c = new BDDDomain[] { makeDomain("H1c", HC_BITS) };
            H2c = new BDDDomain[] { makeDomain("H2c", HC_BITS) };
        } else {
            H1c = H2c = new BDDDomain[0];
        }
        */
        
        //if (TRACE) out.println("Variable context domains: "+V1c.length);
        //if (TRACE) out.println("Heap context domains: "+H1c.length);
        
        if (varorder == null) {
            // default variable orderings.
            /*
            if (CONTEXT_SENSITIVE || THREAD_SENSITIVE || OBJECT_SENSITIVE) {
                if (HC_BITS > 0) {
                    varorder = "N_F_Z_I_M2_M_T1_V2xV1_V2cxV1c_H2xH2c_T2_H1xH1c";
                } else {
                    //varorder = "N_F_Z_I_M2_M_T1_V2xV1_V2cxV1c_H2_T2_H1";
                    varorder = "N_F_I_M2_M_Z_V2xV1_V2cxV1c_T1_H2_T2_H1";
                }
            } else if (CARTESIAN_PRODUCT && false) {
                varorder = "N_F_Z_I_M2_M_T1_V2xV1_T2_H2xH1";
                for (int i = 0; i < V1c.length; ++i) {
                    varorder += "xV1c"+i+"xV2c"+i;
                }
            } else {
                //varorder = "N_F_Z_I_M2_M_T1_V2xV1_H2_T2_H1";
                varorder = "N_F_I_M2_M_Z_V2xV1_T1_H2_T2_H1";
            } */
            varorder = "F_I_M_Z_V2xV1_T1_T2_H1";
        }
        
        System.out.println("Using variable ordering "+varorder);
        int[] ordering = bdd.makeVarOrdering(reverseLocal, varorder);
        bdd.setVarOrder(ordering);
        
        /*
        V1ctoV2c = bdd.makePair();
        V1ctoV2c.set(V1c, V2c);
        V1cV2ctoV2cV1c = bdd.makePair();
        V1cV2ctoV2cV1c.set(V1c, V2c);
        V1cV2ctoV2cV1c.set(V2c, V1c);
        if (OBJECT_SENSITIVE) {
            V1cH1ctoV2cV1c = bdd.makePair();
            V1cH1ctoV2cV1c.set(V1c, V2c);
            V1cH1ctoV2cV1c.set(H1c, V1c);
        }
        T2toT1 = bdd.makePair(T2, T1);
        T1toT2 = bdd.makePair(T1, T2);
        V1toV2 = bdd.makePair();
        V1toV2.set(V1, V2);
        V1toV2.set(V1c, V2c);
        V2toV1 = bdd.makePair();
        V2toV1.set(V2, V1);
        V2toV1.set(V2c, V1c);
        H1toH2 = bdd.makePair();
        H1toH2.set(H1, H2);
        H1toH2.set(H1c, H2c);
        H2toH1 = bdd.makePair();
        H2toH1.set(H2, H1);
        H2toH1.set(H2c, H1c);
        V1H1toV2H2 = bdd.makePair();
        V1H1toV2H2.set(V1, V2);
        V1H1toV2H2.set(H1, H2);
        V1H1toV2H2.set(V1c, V2c);
        V1H1toV2H2.set(H1c, H2c);
        V2H2toV1H1 = bdd.makePair();
        V2H2toV1H1.set(V2, V1);
        V2H2toV1H1.set(H2, H1);
        V2H2toV1H1.set(V2c, V1c);
        V2H2toV1H1.set(H2c, H1c);
        */
        
        V1set = V1.set();
        /*
        if (V1c.length > 0) {
            V1cset = bdd.one();
            V1cdomain = bdd.one();
            for (int i = 0; i < V1c.length; ++i) {
                V1cset.andWith(V1c[i].set());
                V1cdomain.andWith(V1c[i].domain());
            }
            V1set.andWith(V1cset.id());
        }
        */
        V2set = V2.set();
        /*
        if (V2c.length > 0) {
            V2cset = bdd.one();
            V2cdomain = bdd.one();
            for (int i = 0; i < V2c.length; ++i) {
                V2cset.andWith(V2c[i].set());
                V2cdomain.andWith(V2c[i].domain());
            }
            V2set.andWith(V2cset.id());
        }
        */
        H1set = H1.set();
        /*
        if (H1c.length > 0) {
            H1cset = bdd.one();
            H1cdomain = bdd.one();
            for (int i = 0; i < H1c.length; ++i) {
                H1cset.andWith(H1c[i].set());
                H1cdomain.andWith(H1c[i].domain());
            }
            H1set.andWith(H1cset.id());
        }
        H2set = H2.set();
        if (H2c.length > 0) {
            H2cset = bdd.one();
            H2cdomain = bdd.one();
            for (int i = 0; i < H2c.length; ++i) {
                H2cset.andWith(H2c[i].set());
                H2cdomain.andWith(H2c[i].domain());
            }
            H2set.andWith(H2cset.id());
        }
        */
        T1set = T1.set();
        T2set = T2.set();
        Fset = F.set();
        Mset = M.set();
        //Nset = N.set();
        Iset = I.set();
        Zset = Z.set();
        /*
        V1cV2cset = (V1c.length > 0) ? V1cset.and(V2cset) : bdd.zero();
        H1cH2cset = (H1c.length > 0) ? H1cset.and(H2cset) : bdd.zero();
        if (V1c.length > 0) {
            V1cH1cset = (H1c.length > 0) ? V1cset.and(H1cset) : V1cset;
        } else {
            V1cH1cset = (H1c.length > 0) ? H1cset : bdd.zero();
        }*/
        V1V2set = V1set.and(V2set);
        V1FV2set = V1V2set.and(Fset);
        V1H1set = V1set.and(H1set);
        V1Fset = V1set.and(Fset);
        V2Fset = V2set.and(Fset);
        IMset = Iset.and(Mset);
        //INset = Iset.and(Nset);
        //INH1set = INset.and(H1set);
        //INT2set = INset.and(T2set);
        H1Fset = H1set.and(Fset);
        //H2Fset = H2set.and(Fset);
        //H1H2set = H1set.and(H2set);
        //H1FH2set = H1Fset.and(H2set);
        //T2Nset = T2set.and(Nset);
        MZset = Mset.and(Zset);
        
        A = bdd.zero();
        vP = bdd.zero();
        S = bdd.zero();
        L = bdd.zero();
        vT = bdd.zero();
        hT = bdd.zero();
        aT = bdd.zero();
        /*
        if (FILTER_HP) {
            fT = bdd.zero();
            fC = bdd.zero();
        }
        cha = bdd.zero();
        */
        actual = bdd.zero();
        formal = bdd.zero();
        Iret = bdd.zero();
        Mret = bdd.zero();
        Ithr = bdd.zero();
        Mthr = bdd.zero();
        //mI = bdd.zero();
        //mV = bdd.zero();
        //sync = bdd.zero();
        IE = bdd.zero();
        //hP = bdd.zero();
        //visited = bdd.zero();
        /*
        if (OBJECT_SENSITIVE || CARTESIAN_PRODUCT) staticCalls = bdd.zero();
        
        if (THREAD_SENSITIVE) threadRuns = bdd.zero();
        
        if (INCREMENTAL1) {
            old1_A = bdd.zero();
            old1_S = bdd.zero();
            old1_L = bdd.zero();
            old1_vP = bdd.zero();
            old1_hP = bdd.zero();
        }
        if (INCREMENTAL2) {
            old2_myIE = bdd.zero();
            old2_visited = bdd.zero();
        }
        if (INCREMENTAL3) {
            old3_t3 = bdd.zero();
            old3_vP = bdd.zero();
            old3_t4 = bdd.zero();
            old3_hT = bdd.zero();
            old3_t6 = bdd.zero();
            old3_t9 = new BDD[MAX_PARAMS];
            for (int i = 0; i < old3_t9.length; ++i) {
                old3_t9[i] = bdd.zero();
            }
        }
        
        if (CARTESIAN_PRODUCT && false) {
            H1toV1c = new BDDPairing[MAX_PARAMS];
            V1ctoH1 = new BDDPairing[MAX_PARAMS];
            V1csets = new BDD[MAX_PARAMS];
            V1cH1equals = new BDD[MAX_PARAMS];
            for (int i = 0; i < MAX_PARAMS; ++i) {
                H1toV1c[i] = bdd.makePair(H1, V1c[i]);
                V1ctoH1[i] = bdd.makePair(V1c[i], H1);
                V1csets[i] = V1c[i].set();
                V1cH1equals[i] = H1.buildEquals(V1c[i]);
            }
        }
        
        if (USE_VCONTEXT) {
            IEcs = bdd.zero();
        }
        */
    }
    
    
    //static int nextID = 0;
    private class ASTNodeWrapper { 
        //int id; // unique id
        ASTNode n; // null for global this
        boolean isStatic;
        //int scope;
        
        ASTNodeWrapper(ASTNode obj, boolean isStat) {
            // peel off parens
            if (obj != null && 
                obj.getNodeType() == ASTNode.PARENTHESIZED_EXPRESSION) {
                n = ((ParenthesizedExpression) obj).getExpression();
            }
            else {
                n = obj; 
            }
            isStatic = isStat;
            //id = nextID++;
            //scope = s;
        }
        ASTNodeWrapper(ASTNode obj) {
            this(obj, false);
        }
        
        public String toString() {
            if (n == null) return "NODE type: " + "global this";
            return "NODE type: " + n.getNodeType() + /* ", id: " + id 
                        + ", scope: " + scope + */", " + n.toString();
        }
        
        public boolean equals(Object o) {
            if (o instanceof ASTNodeWrapper) {
                ASTNodeWrapper rhs = (ASTNodeWrapper) o;
                ASTNode m = rhs.n;
                if (m == n) {
                    System.out.println("NODE equals: " 
                        + m.toString() +" == "+ n.toString());
                    return true;
                }
                if (m == null || n == null) return false;
                if (m.getAST() != n.getAST()) {
                    System.out.println("m.AST != n.AST");
                    return false;
                }
                switch (m.getNodeType()) {
                    case ASTNode.PARENTHESIZED_EXPRESSION:
                        return equals(new 
                            ASTNodeWrapper(((ParenthesizedExpression) m).getExpression()));
                    case ASTNode.SIMPLE_NAME:
                    case ASTNode.QUALIFIED_NAME:
                        // TODO must resolve bindings to determine if it can be true
                        if (n instanceof Name) {
                            return ((Name) n).getFullyQualifiedName().equals(((Name) m).getFullyQualifiedName());
                        }
                        return false;
                    case ASTNode.CLASS_INSTANCE_CREATION:
                    case ASTNode.ARRAY_CREATION: 
                    case ASTNode.STRING_LITERAL:
                        System.out.println("NODE equals: " 
                                + m.toString() +" != "+ n.toString());
                        return false; // since m != n
                    // TODO not complete, other types to be handled
                }                
                return false;
            }
            return false;
        }
        
        public int hashCode() {
            if (n == null) return 0;
            return n.hashCode();
        }
    }
    
    
    /**
     * @author jimz
     */
    private class PAASTVisitor extends ASTVisitor { 
        /*
        public PAASTVisitor(int i, int s) {
            super(false); // avoid comments
            id = i;
            scope = s;
        }
         */
        public PAASTVisitor() { super(false); /*this(0,0);*/};
        
        // vP
        public boolean visit(ArrayCreation arg0) {
            ASTNodeWrapper node = new ASTNodeWrapper(arg0);
            addToVP(node, node);
            return true;
        }
        public boolean visit(ClassInstanceCreation arg0) {
            ASTNodeWrapper node = new ASTNodeWrapper(arg0);
            addToVP(node, node);
            return true;
        }
        public boolean visit(StringLiteral arg0) {
            // XXX should i unify equivalent string literals?
            ASTNodeWrapper node = new ASTNodeWrapper(arg0);
            addToVP(node, node);          
            return true;
        }
        
        // A
        public boolean visit(Assignment arg0) {
            return true; // add to A after visit
        }
        public void endVisit(Assignment arg0) {
            // TODO need to resolve bindings
            Expression right = arg0.getRightHandSide();  
            ITypeBinding rType = right.resolveTypeBinding();
            Expression left = arg0.getLeftHandSide();
            if (left.getNodeType() == ASTNode.PARENTHESIZED_EXPRESSION) {
                left = ((ParenthesizedExpression) left).getExpression();
            }
            if (rType == null) { // bindings not available
                // TODO check types?
                compareAssignment(left, right);
            }
            else {              
                if (rType.isPrimitive()) return;
                if (rType.isNullType()) {
                    // XXX what to do when null?
                }
                else {
                    compareAssignment(left, right);
                }
            } 
        }
        
        // formal
        public boolean visit(MethodDeclaration arg0) {
            // TODO add this?
 
            int modifiers = arg0.getModifiers();
            ASTNodeWrapper thisparam;
            if (Modifier.isStatic(modifiers)) {
                thisparam = new ASTNodeWrapper(null, true);
            }
            else {
                ThisExpression t = arg0.getAST().newThisExpression();
                t.setQualifier(arg0.getName());
                thisparam = new ASTNodeWrapper(t);
            }
            
            int M_i = Mmap.get(new ASTNodeWrapper(arg0.getName()));
            BDD M_bdd = M.ithVar(M_i);
            
            addToFormal(M_bdd, 0, thisparam);
            
            List params = arg0.parameters();
            Iterator it = params.iterator();
            for (int i = 1; it.hasNext(); i++) {
                SingleVariableDeclaration v = (SingleVariableDeclaration)it.next();
                addToFormal(M_bdd, i, new ASTNodeWrapper(v.getName()));
            }
            
            // throws, returns?
            
            arg0.getBody().accept(this);
            
            return false; // do not go into the decls
        }
        
        public boolean visit(SingleVariableDeclaration arg0) {
            // need to add to map first and declare static or not
            Vmap.get(new ASTNodeWrapper(arg0.getName(), 
                Modifier.isStatic(arg0.getModifiers())));
            // TODO: add assignment
            return true;
        }
        
        public boolean visit(VariableDeclarationStatement arg0) {
            boolean isStatic = Modifier.isStatic(arg0.getModifiers());
            addToVmap(arg0.fragments(), isStatic);
            return true;
        }
        
        public boolean visit(FieldDeclaration arg0) {
            boolean isStatic = Modifier.isStatic(arg0.getModifiers());
            addToVmap(arg0.fragments(), isStatic);
            return true;
        }
        
        private void compareAssignment(Expression left, Expression right) {
            switch (right.getNodeType()) {
                case ASTNode.PARENTHESIZED_EXPRESSION:
                    compareAssignment(left, 
                        ((ParenthesizedExpression) right).getExpression());
                    break;
                case ASTNode.ASSIGNMENT:    
                    addToA(left, ((Assignment) right).getLeftHandSide());
                    break;
                case ASTNode.CLASS_INSTANCE_CREATION:
                case ASTNode.ARRAY_CREATION:
                case ASTNode.STRING_LITERAL:
                    addToA(left, right);
                    break;  
                case ASTNode.SIMPLE_NAME:
                    // FIXME this is broken without bindings
                    // and need to distinguish between loads/stores/assignments
                    addToA(left, right);
                    break;
                case ASTNode.QUALIFIED_NAME:
                    
                    break;         
                case ASTNode.THIS_EXPRESSION:
         
                    break;     
                case ASTNode.SUPER_FIELD_ACCESS:
                    
                    break;                                  
                case ASTNode.FIELD_ACCESS:
        
                    break;                    
                case ASTNode.NULL_LITERAL:
                    // XXX what to do for null?
                    break;

                case ASTNode.METHOD_INVOCATION:
                    
                    break;
                case ASTNode.SUPER_METHOD_INVOCATION:
                           
                    break;                
                case ASTNode.ARRAY_ACCESS:
                    
                    break;
                case ASTNode.INFIX_EXPRESSION:
                    // must be string
                    break;
                case ASTNode.CONDITIONAL_EXPRESSION:
                    
                    break;     
                case ASTNode.CAST_EXPRESSION:
                        
                    break;
                default:
                    // e.g. postfixexpression, do nothing
            }
        }

        public void postVisit(ASTNode arg0) {
            // TODO Auto-generated method stub
        }
        public void preVisit(ASTNode arg0) {
            // TODO Auto-generated method stub
        }
        public boolean visit(AnonymousClassDeclaration arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(ArrayAccess arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(Block arg0) {
            //scope++;
            return true;
        }
        /*public void endVisit(Block arg0) {
            //scope--;            
        }*/
        public boolean visit(BooleanLiteral arg0) {
            return true;
        }
        public boolean visit(BreakStatement arg0) {
            return true;
        }
        public boolean visit(CastExpression arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(CatchClause arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(CharacterLiteral arg0) {
            return true;
        }
        public boolean visit(CompilationUnit arg0) {
            return true; 
        }
        public boolean visit(ConditionalExpression arg0) {
            // TODO: count as assignment
            return true; 
        }
        public boolean visit(ConstructorInvocation arg0) {
            // TODO Auto-generated method stub
            return true; 
        }
        public boolean visit(FieldAccess arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(InfixExpression arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(InstanceofExpression arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(MethodInvocation arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(Modifier arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(NullLiteral arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(QualifiedName arg0) {
            // docs say this might also be a field access
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(ReturnStatement arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(SimpleName arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(SimpleType arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(SuperConstructorInvocation arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(SuperFieldAccess arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(SuperMethodInvocation arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(ThisExpression arg0) {
            // assign to resolved expr 
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(ThrowStatement arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(TypeDeclaration arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(TypeDeclarationStatement arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(TypeLiteral arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(VariableDeclarationExpression arg0) {
            // TODO Auto-generated method stub
            // used only in for loops
            // only final modifier here, no need for static check
            return true;
        }
        public boolean visit(VariableDeclarationFragment arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        
        // empty visitors
        public boolean visit(ArrayInitializer arg0) {
            return true;
        }
        public boolean visit(ArrayType arg0) {
            return true;
        }
        public boolean visit(AssertStatement arg0) {
            return true;
        }
        public boolean visit(Initializer arg0) {
            return true;
        }
        public boolean visit(LabeledStatement arg0) {
            return true;
        }
        public boolean visit(ContinueStatement arg0) {
            return true; 
        }
        public boolean visit(DoStatement arg0) {
            return true; 
        }
        public boolean visit(ExpressionStatement arg0) {
            return true;
        }
        public boolean visit(ForStatement arg0) {
            return true;
        }
        public boolean visit(IfStatement arg0) {
            return true;
        }
        public boolean visit(ImportDeclaration arg0) {
            return true;
        }
        public boolean visit(NumberLiteral arg0) {
            return true;
        }
        public boolean visit(PackageDeclaration arg0) {
            return true;
        }
        public boolean visit(ParenthesizedExpression arg0) {
            return true;
        }
        public boolean visit(PostfixExpression arg0) {
            return true;
        }
        public boolean visit(PrefixExpression arg0) {
            return true;
        }
        public boolean visit(PrimitiveType arg0) {
            return true;
        }
        public boolean visit(SwitchCase arg0) {
            return true;
        }
        public boolean visit(SwitchStatement arg0) {
            return true;
        }
        public boolean visit(SynchronizedStatement arg0) {
            return true;
        }
        public boolean visit(TryStatement arg0) {
            return true;
        }
        public boolean visit(WhileStatement arg0) {
            return true;
        }     
        
        /* TODO: JLS3
        public boolean visit(EnhancedForStatement arg0) {
            // TODO Auto-generated method stub
            return true; 
        }
        public boolean visit(EnumConstantDeclaration arg0) {
            // TODO Auto-generated method stub
            return true; 
        }
        public boolean visit(EnumDeclaration arg0) {
            // TODO Auto-generated method stub
            return true; 
        }              
        public boolean visit(AnnotationTypeDeclaration arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(AnnotationTypeMemberDeclaration arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(MarkerAnnotation arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(MemberValuePair arg0) {
            // TODO Auto-generated method stub
            return true;
        }        
        public boolean visit(NormalAnnotation arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        public boolean visit(ParameterizedType arg0) {
            // TODO Auto-generated method stub
            return true;
        }        
        public boolean visit(QualifiedType arg0) {
            // TODO Auto-generated method stub
            return true;
        }        
        public boolean visit(SingleMemberAnnotation arg0) {
            // TODO Auto-generated method stub
            return true;
        }        
        public boolean visit(TypeParameter arg0) {
            // TODO Auto-generated method stub
            return true;
        }        
        public boolean visit(WildcardType arg0) {
            // TODO Auto-generated method stub
            return true;
        }
        */
    }
   
    
    // Read in default properties.
    static { SystemProperties.read("pas.properties"); }
    
    static boolean USE_JOEQ_CLASSLIBS = !System.getProperty("pas.usejoeqclasslibs", "no").equals("no");

    //Set/*<CompilationUnit>*/ ast;
    List/*<CompilationUnit>*/ todo;
    
    public PAFromSource() {
        //ast = new HashSet();
        todo = new ArrayList();
    }
    
    public static void main(String[] args) throws IOException {
        // is this stuff necessary?
        if (USE_JOEQ_CLASSLIBS) {
            System.setProperty("joeq.classlibinterface", "joeq.ClassLib.pas.Interface");
            joeq.ClassLib.ClassLibInterface.useJoeqClasslib(true);
        }
        HostedVM.initialize();
        
        List files;    
        if (args[0].startsWith("@")) {
            files = readClassesFromFile(args[0].substring(1));
        } else {
            files = Collections.singletonList(args[0]);
        }
        
        PAFromSource dis = new PAFromSource();
     
        dis.run(files);
    }

    
    static List/*<String>*/ readClassesFromFile(String fname) 
        throws IOException {
        
        BufferedReader r = null;
        try {
            r = new BufferedReader(new FileReader(fname));
            List classes = new ArrayList();
            String s = null;
            while ((s = r.readLine()) != null) {
                classes.add(s);
            }
            return classes;
        } finally {
            if (r != null) r.close();
        }
    }
    
    
    /**
     * @param files
     * @throws IOException
     */
    void run(List files) throws IOException {
        initializeBDD("java");
        initializeMaps();

        generateASTs(files);
        
        // Start timing.
        long time = System.currentTimeMillis();
        
        while (!todo.isEmpty()) {
            generateRelations();
        }
        
        System.out.println("Time spent generating relations: "+(System.currentTimeMillis()-time)/1000.);
               
        System.out.println("Writing relations...");
        time = System.currentTimeMillis();
        dumpBDDRelations();
        System.out.println("Time spent writing: "+(System.currentTimeMillis()-time)/1000.);
    }

    public void dumpBDDRelations() throws IOException {
        
        // difference in compatibility
        BDD S0 = S;//.exist(V1cV2cset);
        BDD L0 = L;//.exist(V1cV2cset);
        BDD IE0 = IE;//.exist(V1cV2cset);
        BDD vP0 = vP;//vP.exist(V1cH1cset);
        
        String dumpPath = System.getProperty("pas.dumppath", "");
        if (dumpPath.length() > 0) {
            File f = new File(dumpPath);
            if (!f.exists()) f.mkdirs();
            String sep = System.getProperty("file.separator", "/");
            if (!dumpPath.endsWith(sep))
                dumpPath += sep;
        }
        System.out.println("Dumping to path "+dumpPath);
        
        DataOutputStream dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"bddinfo"));
            for (int i = 0; i < bdd.numberOfDomains(); ++i) {
                BDDDomain d = bdd.getDomain(i);
                if (d == V1 || d == V2)
                    dos.writeBytes("V\n");
                else if (d == H1)// || d == H2)
                    dos.writeBytes("H\n");
                else if (d == T1 || d == T2)
                    dos.writeBytes("T\n");
                else if (d == F)
                    dos.writeBytes("F\n");
                else if (d == I)
                    dos.writeBytes("I\n");
                else if (d == Z)
                    dos.writeBytes("Z\n");
                /*else if (d == N)
                    dos.writeBytes("N\n");*/
                else if (d == M)// || d == M2)
                    dos.writeBytes("M\n");
                /*else if (Arrays.asList(V1c).contains(d)
                        || Arrays.asList(V2c).contains(d))
                    dos.writeBytes("VC\n");
                else if (Arrays.asList(H1c).contains(d)
                        || Arrays.asList(H2c).contains(d))
                    dos.writeBytes("HC\n");
                else if (DUMP_SSA) {
                    dos.writeBytes(bddIRBuilder.getDomainName(d)+"\n");
                }*/
                else
                    dos.writeBytes(d.toString() + "\n");
            }
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"fielddomains.pa"));
            dos.writeBytes("V "+(1L<<V_BITS)+" var.map\n");
            dos.writeBytes("H "+(1L<<H_BITS)+" heap.map\n");
            dos.writeBytes("T "+(1L<<T_BITS)+" type.map\n");
            dos.writeBytes("F "+(1L<<F_BITS)+" field.map\n");
            dos.writeBytes("I "+(1L<<I_BITS)+" invoke.map\n");
            dos.writeBytes("Z "+(1L<<Z_BITS)+"\n");
            //dos.writeBytes("N "+(1L<<N_BITS)+" name.map\n");
            dos.writeBytes("M "+(1L<<M_BITS)+" method.map\n");
            //dos.writeBytes("VC "+(1L<<VC_BITS)+"\n");
            //dos.writeBytes("HC "+(1L<<HC_BITS)+"\n");
            //if (bddIRBuilder != null) bddIRBuilder.dumpFieldDomains(dos);
        } finally {
            if (dos != null) dos.close();
        }
        /*
        BDD mC = bdd.zero();
        for (Iterator i = visited.iterator(Mset); i.hasNext(); ) {
            BDD m = (BDD) i.next();
            int m_i = (int) m.scanVar(M);
            jq_Method method = (jq_Method) Mmap.get(m_i);
            BDD c = getV1V2Context(method);
            if (c != null) {
                BDD d = c.exist(V2cset); c.free();
                m.andWith(d);
            }
            mC.orWith(m);
        }
        */
        bdd.save(dumpPath+"vP0.bdd", vP0);
        //bdd.save(dumpPath+"hP0.bdd", hP);
        bdd.save(dumpPath+"L.bdd", L0);
        bdd.save(dumpPath+"S.bdd", S0);
        /*if (CONTEXT_SENSITIVE) {
            bdd.save(dumpPath+"cA.bdd", A);
        } else */{
            bdd.save(dumpPath+"A.bdd", A);
        }
        bdd.save(dumpPath+"vT.bdd", vT);
        bdd.save(dumpPath+"hT.bdd", hT);
        bdd.save(dumpPath+"aT.bdd", aT);
        //bdd.save(dumpPath+"cha.bdd", cha);
        bdd.save(dumpPath+"actual.bdd", actual);
        bdd.save(dumpPath+"formal.bdd", formal);
        //bdd.save(dumpPath+"mV.bdd", mV);
        //bdd.save(dumpPath+"mC.bdd", mC);
        //bdd.save(dumpPath+"mI.bdd", mI);
        bdd.save(dumpPath+"Mret.bdd", Mret);
        bdd.save(dumpPath+"Mthr.bdd", Mthr);
        bdd.save(dumpPath+"Iret.bdd", Iret);
        bdd.save(dumpPath+"Ithr.bdd", Ithr);
        bdd.save(dumpPath+"IE0.bdd", IE0);
        //bdd.save(dumpPath+"sync.bdd", sync);
        /*if (threadRuns != null)
            bdd.save(dumpPath+"threadRuns.bdd", threadRuns);
        if (IEfilter != null)
            bdd.save(dumpPath+"IEfilter.bdd", IEfilter);
        bdd.save(dumpPath+"roots.bdd", getRoots());

        if (V1c.length > 0 && H1c.length > 0) {
            bdd.save(dumpPath+"eq.bdd", V1c[0].buildEquals(H1c[0]));
        }
        
        if (DUMP_FLY) {
            initFly();
            bdd.save(dumpPath+"visited.bdd", visitedFly);
            bdd.save(dumpPath+"mS.bdd", mS);
            bdd.save(dumpPath+"mL.bdd", mL);
            bdd.save(dumpPath+"mvP.bdd", mvP);
            bdd.save(dumpPath+"mIE.bdd", mIE);
        }
        */
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"var.map"));
            Vmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"heap.map"));
            Hmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"type.map"));
            Tmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"field.map"));
            Fmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"invoke.map"));
            Imap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"name.map"));
            Nmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }
        
        dos = null;
        try {
            dos = new DataOutputStream(new FileOutputStream(dumpPath+"method.map"));
            Mmap.dumpStrings(dos);
        } finally {
            if (dos != null) dos.close();
        }

    }
    
    IndexMap makeMap(String name, int bits) {
        return new IndexMap(name, 1 << bits);
    }

    private void initializeMaps() {
        Vmap = makeMap("Vars", V_BITS);
        Imap = makeMap("Invokes", I_BITS);
        Hmap = makeMap("Heaps", H_BITS);
        Fmap = makeMap("Fields", F_BITS);
        Tmap = makeMap("Types", T_BITS);
        Nmap = makeMap("Names", N_BITS);
        Mmap = makeMap("Methods", M_BITS);
        Mmap.get("DUMMY object"); // XXX should use PA.Dummy instead?
    }

    private void generateASTs(List files) {
        long time = System.currentTimeMillis();
        
        for (Iterator i = files.iterator(); i.hasNext(); ) {
            CompilationUnit cu = readSourceFile((String) i.next());
            todo.add(cu);
        }
    
        System.out.println("Time spent parsing: "+(System.currentTimeMillis()-time)/1000.);
    }
    
    
    private void generateRelations() {
        CompilationUnit cu = (CompilationUnit)todo.remove(todo.size()-1);
        
        cu.accept(new PAASTVisitor());  
        
    }
    
    static CompilationUnit readSourceFile(String fname) {        
        System.out.print("parsing file: " + fname);
 
        StringBuffer sb = new StringBuffer();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fname));
            int c;
            while ((c = br.read()) != -1) {
                sb.append((char) c);
            }
            br.close();
        }
        catch (IOException e) {
            System.out.println(" ... error opening file. Exiting.");
            System.exit(1);
        }
        
        char[] source = new char[sb.length()];
        sb.getChars(0, sb.length(), source, 0);
        
        //if (sb.length() > 100) System.out.println(sb.substring(0,100)); // remove later
        
        ASTParser parser = ASTParser.newParser(AST.JLS2); // = ASTParser.newParser(AST.JLS3);
        parser.setResolveBindings(true);
        parser.setUnitName(fname);
        parser.setSource(source); 
        //parser.setResolveBindings(true);
        CompilationUnit cu = (CompilationUnit)parser.createAST(null);
        if (cu.getMessages().length == 0) {
            System.out.println(" ... complete."); 
        }
        else {
            System.out.println(" ... parse error. Exiting.");
            System.exit(1);
        }
        
        return cu;
    }
    
    void addToVP(ASTNodeWrapper v, ASTNodeWrapper h) {
        int V_i = Vmap.get(v);
        int H_i = Hmap.get(h);
        BDD V_bdd = V1.ithVar(V_i);
        BDD bdd1 = H1.ithVar(H_i);
        bdd1.andWith(V_bdd); // .id()?
        out.println("adding to vP " + v + " / " + h); 
        if (TRACE_RELATIONS) out.println("Adding to vP: "+bdd1.toStringWithDomains());
        vP.orWith(bdd1);
    }
    
    void addToA(ASTNode v1, ASTNode v2) {
        int V1_i = Vmap.get(new ASTNodeWrapper(v1));
        int V2_i = Vmap.get(new ASTNodeWrapper(v2));        
        BDD V_bdd = V1.ithVar(V1_i);        
        BDD bdd1 = V2.ithVar(V2_i);
        bdd1.andWith(V_bdd);// .id()?
        out.println("adding to A " + v1 + " / " + v2); 
        if (TRACE_RELATIONS) out.println("Adding to A: "+bdd1.toStringWithDomains());
        A.orWith(bdd1);
    }
    
    void addToFormal(BDD M_bdd, int z, ASTNodeWrapper v) {
        BDD bdd1 = Z.ithVar(z);
        int V_i = Vmap.get(v);
        bdd1.andWith(V1.ithVar(V_i));
        bdd1.andWith(M_bdd.id());
        if (TRACE_RELATIONS) out.println("Adding to formal: "+bdd1.toStringWithDomains());
        formal.orWith(bdd1);
    }

    private void addToVmap(List frags, boolean isStatic) {
        for (Iterator i = frags.iterator(); i.hasNext(); ) {
            VariableDeclarationFragment vf = 
                (VariableDeclarationFragment) i.next();
            Vmap.get(new ASTNodeWrapper(vf.getName(), isStatic));
        }
    }
}
