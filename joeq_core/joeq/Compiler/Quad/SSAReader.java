/*
 * SSAReader.java
 *
 * Created on November 26, 2002, 3:33 PM
 */

package Compil3r.Quad;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.BufferedReader;
import java.util.zip.GZIPInputStream;
import java.io.IOException;
import java.util.Set;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.LinkedHashSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Collections;
import UTF.Utf8;
import Clazz.jq_NameAndDesc;
import Main.jq;
import Main.HostedVM;

import Compil3r.Quad.AndersenPointerAnalysis;
import Compil3r.BytecodeAnalysis.BytecodeVisitor;
import Compil3r.BytecodeAnalysis.CallTargets;

import Compil3r.Quad.MethodSummary.Node;
import Compil3r.Quad.MethodSummary.ConcreteTypeNode;
import Compil3r.Quad.MethodSummary.ParamNode;
import Compil3r.Quad.MethodSummary.GlobalNode;
import Compil3r.Quad.MethodSummary.ReturnValueNode;
import Compil3r.Quad.AndersenInterface.*;

/**
 *
 * @author  Daniel Wright
 */
public class SSAReader {
    static class SSAClass implements AndersenClass {
        final Utf8 name;
        final String fileLocation;
	final SSAType type;
        ArrayList superclasses;
        HashSet ancestors;
        
        public SSAClass(Utf8 name, String fileLocation) {
            this.name = name;
            this.fileLocation = fileLocation;
	    this.type = new SSAType(this);
        }
        
        public void addSuperclass(SSAClass superclass) {
            if (superclasses == null) superclasses = new ArrayList();
            superclasses.add(superclass);
        }
        
        /**
         * This instructs this class to compute its set of ancestors 
         * of instanceOf computations. It should be called after all classes
         * have been loaded but before instanceOf is called
         */ 
        public HashSet computeAncestors() {
            if (ancestors != null) return ancestors;
            ancestors = new HashSet();
            
            if (superclasses!=null) {
                for (int i=0; i<superclasses.size(); i++) {
                    SSAClass superClass = (SSAClass)superclasses.get(i);
                    ancestors.addAll(superClass.computeAncestors());
                }
            }

            ancestors.add(this);

            return ancestors;
        }
        
        public boolean instanceOf(SSAClass klass) {
            if (ancestors==null)
                throw new IllegalStateException();
            
            return ancestors.contains(klass);
        }
        
        public AndersenClassInitializer and_getClassInitializer() {
            return null;
        }

	public SSAType getType() { return type; }

        public String toString() { return name.toString(); }
        
        public void load() {}; public void verify() {}; public void prepare() {};
    }
    
    abstract static class SSAMember implements AndersenMember {
        final Utf8 name;
        final String fileLocation;
        final SSAClass declaringClass;
        final boolean isstatic;
        
        public SSAMember(Utf8 name, String fileLocation, SSAClass declaringClass, boolean isstatic) {
            this.name = name;
            this.fileLocation = fileLocation;
            this.declaringClass = declaringClass;
            this.isstatic = isstatic;
        }
        
        public SSAClass getDeclaringClass() { return declaringClass; }
        public AndersenClass and_getDeclaringClass() { return declaringClass; }
        public Utf8 getName() { return name; }
        
        public jq_NameAndDesc getNameAndDesc() {
            throw new UnsupportedOperationException();
        }
    }
    
    static class SSAMethod extends SSAMember implements AndersenMethod {
        SSAType[] params;
        ArrayList callTargets;
        MethodSummary summary = null;
	public final boolean isConstructor;
	public final boolean isCompilerGenerated;
        
        public SSAMethod(Utf8 name, String fileLocation, SSAClass declaringClass, 
			 boolean isstatic, boolean isConstructor, boolean isCompilerGenerated) {
            super(name, fileLocation, declaringClass, isstatic);
	    this.isConstructor = isConstructor;
	    this.isCompilerGenerated = isCompilerGenerated;

	    /*
	      Changed - just force the return type to be a concreteType in readprocdef
	    if (isConstructor && isCompilerGenerated) {
		// special case - we make a methodSummary that returns the correct data type
		ParamNode[] paramNodeArray = { new ParamNode(this, 0, universalType) };
		ConcreteTypeNode returned = new ConcreteTypeNode(declaringClass.getType().getPointerToThis());
		summary = new MethodSummary(this, paramNodeArray,
					    new GlobalNode(),
					    Collections.EMPTY_SET,
					    new HashMap(),
					    new HashMap(),
					    Collections.singleton(returned),
					    Collections.EMPTY_SET,
					    Collections.EMPTY_SET);
	    }
	    */
        }
        
        public AndersenType and_getReturnType() {
            throw new UnsupportedOperationException();
        }

        public int getNumParams() {
            if (params==null)
                throw new IllegalStateException();
            
            return params.length;
        }
        
        public SSAType getParamType(int i) {
            if (params==null)
                throw new IllegalStateException();
            
            return params[i];
        }

        public void setParamTypes(SSAType[] params) {
            this.params = params;
        }

        public void addCallTarget(SSAMethod target) {
            if (callTargets==null) callTargets = new ArrayList();
            callTargets.add(target);
        }

        // hack - slow and probably doesn't handle diamond inheritance
        public CallTargets getCallTargets(SSAClass receiverClass, boolean exact) {
            if (callTargets==null) { // non-virtual method, or not overridden
                return new CallTargets.SingleCallTarget(this, true);
            }
            
	    if (receiverClass==null) {
		// receiverClass is universal type - use this method's declaring class
		receiverClass = getDeclaringClass();
		exact = false; // can't have an 'exact' universal type
	    }
	    
            CallTargets ct = CallTargets.NoCallTarget.INSTANCE;
            
            boolean overridden = false;
            
            for (int i=0; i<callTargets.size(); i++) {
                SSAMethod target = (SSAMethod)callTargets.get(i);
                
                if (!receiverClass.instanceOf(target.getDeclaringClass())) {
                    if (exact) continue;
                    
                    if (target.getDeclaringClass().instanceOf(receiverClass)) {
                        ct = ct.union(target.getCallTargets(receiverClass, exact));
                    }
                } else {
                    overridden = true;
                    ct = ct.union(target.getCallTargets(receiverClass, exact));
                }
            }
            
            if (!overridden) ct = ct.union(new CallTargets.SingleCallTarget(this, true));

            return ct;
        }

        public void setSummary(MethodSummary s) { summary = s; }
        public MethodSummary getSummary() { return summary; }
        
        public String toString() {
            return declaringClass.toString()+"."+name.toString();
        }
        
        public boolean isBodyLoaded() {
            return summary != null;
        }
        
    }
    
    static class SSAField extends SSAMember implements AndersenField {
        public SSAField(Utf8 name, String fileLocation, SSAClass declaringClass, boolean isstatic) {
            super(name, fileLocation, declaringClass, isstatic);
        }
        
        public AndersenType and_getType() {
            throw new UnsupportedOperationException();
        }
    }
    
    static class SSAType implements AndersenReference {
        final SSAClass klass;
        final SSAType pointerTargetType;
	final boolean isReference;

	SSAType pointerToThis = null;
	SSAType referenceToThis = null;

        SSAType(SSAClass klass) { 
	    // base type - not a pointer or reference
	    this.klass = klass; 
	    this.pointerTargetType = null;
	    this.isReference = false;
	}

	SSAType(SSAType target, boolean isReference) {
	    // a pointer or reference (pointer => isReference=false)
	    this.klass = null;
	    this.pointerTargetType = target;
	    this.isReference = isReference;
	}
        
	public boolean isBaseType() { return klass!=null; }
	public boolean isPointerType() { return pointerTargetType!=null&&!isReference; }
        public boolean isArrayType() { return isPointerType(); }
	public boolean isReferenceType() { return isReference; }
	public boolean isUniversalType() { return klass==null && pointerTargetType==null; }

        public SSAClass getSSAClass() { return klass; }
	public SSAType getTargetType() { return pointerTargetType; }
	
	public SSAType getPointerToThis(int pointerCount) {
	    if (pointerCount==0) return this;

	    if (pointerToThis==null) {
		pointerToThis = new SSAType(this, false);
	    }

	    return pointerToThis.getPointerToThis(pointerCount-1);
	}
        public SSAType getPointerToThis() { return getPointerToThis(1); }

	public SSAType getReferenceToThis() {
	    if (referenceToThis==null) {
		referenceToThis = new SSAType(this, true);
	    }

	    return referenceToThis;
	}

        public String toString() {
            if (klass!=null) return klass.toString();
	    if (pointerTargetType != null) {
		String targ = pointerTargetType.toString();
		if (isReference)
		    return targ+'&';
		else
		    return targ+'*';
	    }
	    return "UniversalType";
        }
        
        public void load() {}; public void verify() {}; public void prepare() {};
    }
    final static SSAType universalType = new SSAType(null); // as long as everything has the same type, only make one instance
    
    static class LocalVar {
        final Utf8 name;
        final SSAType type;
        ArrayList vvNodes;
        
        public LocalVar(Utf8 name, SSAType type) 
            { this.name = name; this.type = type; vvNodes = new ArrayList(); }

        public void addVVNode(VVNode vvNode) { vvNodes.add(vvNode); }
        public SSAType getType() { return type; }
    }
    
    static class VVNode {
        final LocalVar localVar;
        final ConcreteTypeNode stackVarNode;
        
        public VVNode(LocalVar localVar) { 
            this.localVar = localVar; 
            stackVarNode = new ConcreteTypeNode(localVar.getType());
        }
        public LocalVar getLocalVar() { return localVar; }
        public ConcreteTypeNode getNode() { return stackVarNode; }
    }

    static String[] class_triples =  { "Class" };
    static String[] proc_triples = { "ExternProc", "StaticProc" };
    static String[] proc_quads = { "InstanceMethod", "StaticMethod" };
    static String[] field_quads = { "InstanceMember", "StaticMember" };

    ArrayList classes;
    ArrayList fields;
    ArrayList methods;
    SSAClass globalClass;
    int programLocationCounter;
    
    static void setArrayListElem(ArrayList arrayList, int index, Object element) {
        while (arrayList.size()<index+1) arrayList.add(null);
        arrayList.set(index, element);
    }
    
    static String startsWithElem(String line, String[] prefixes) {
        for (int i=0; i<prefixes.length; i++)
               if (line.startsWith(prefixes[i])) return prefixes[i];
        return null;
    }
    
    static int getIndex(String line, int beginPos, String prefix) {
        if (!line.startsWith(prefix, beginPos)) {
            System.out.println("Error - expected \""+prefix+"\" at position "+beginPos+" in: "+line);
            throw new IllegalArgumentException();
        }
        
        int open = beginPos+prefix.length();
        int close = line.indexOf(']', open+1);
        if (line.charAt(open)!='[' || line.charAt(close)!=']') throw new IllegalArgumentException();
        
        return Integer.parseInt(line.substring(open+1, close));
    }
    
    boolean readClass(String line) {

        // check if this is a class line
        String prefix = startsWithElem(line, class_triples);
        if (prefix == null) return false;
        if (prefix != "Class") {
            System.out.println("Unrecognized: "+line);
            return true;
        }
        
        // extract class number (Class[%d]:....)
        int colon = line.indexOf(':');
        int classNum = Integer.parseInt(line.substring(prefix.length()+1, colon-1));
        
        // extract file location (final field)
        int filelocStart = line.lastIndexOf("{");
        String fileloc = line.substring(filelocStart+1, line.length()-1);

        //finally, extract the name
        String name = line.substring(line.indexOf('"')+1, line.lastIndexOf('"',filelocStart));

        setArrayListElem(classes, classNum, new SSAClass(Utf8.get(name), fileloc));
        
        return true;
    }
    
    boolean readSuperclass(String line) {
        if (!line.startsWith("SUPERCLASS")) return false;
        
        int open1 = line.indexOf('[');
        int open2 = line.indexOf('[',open1+1);
        int close1 = line.indexOf(']');
        int close2 = line.indexOf(']',close1+1);
        
        int baseclass = Integer.parseInt(line.substring(open1+1, close1));
        int superclass = Integer.parseInt(line.substring(open2+1, close2));

        ((SSAClass)classes.get(baseclass)).addSuperclass((SSAClass)classes.get(superclass));

        return true;
    }
    
    boolean readMember(String line) {
        String prefix;
        if (line.startsWith("Field")) prefix="Field";
        else if (line.startsWith("Proc")) prefix="Proc";
        else return false;
        
        boolean isProc = prefix.equals("Proc");
     
        int colon = line.indexOf(':');
        int colon2 = line.indexOf(':', colon+1);
        int number = Integer.parseInt(line.substring(prefix.length()+1, colon-1));
        
        String type = line.substring(colon+2, colon2);
  
        // extract file location (final field)
        int filelocStart = line.lastIndexOf("{");
        String fileLocation = line.substring(filelocStart+1, line.length()-1);

        // extract the name
        String name = line.substring(line.indexOf('"')+1, line.lastIndexOf('"',filelocStart));

	// extract attributes
	StringTokenizer st = new StringTokenizer(line.substring(line.lastIndexOf('[')+1, 
								line.lastIndexOf(']')), ", ");

	boolean isConstructor = false;
	boolean isCompilerGenerated = false;
	while (st.hasMoreTokens()) {
	    String token = st.nextToken();
	    if (token.equals("constructor")) isConstructor = true;
	    if (token.equals("compiler_generated")) isCompilerGenerated = true;
	}

        boolean isglobal = false;
        boolean isstatic = false;
        if (isProc) {
            if (type.equals("ExternProc")) {
                isglobal = true;
                isstatic = true;
            } else if (type.equals("StaticProc")) {
                isglobal = true;
                isstatic = true;
            } else if (type.equals("InstanceMethod")) {
                isglobal = false;
                isstatic = false;
            } else if (type.equals("StaticMethod")) {
                isglobal = false;
                isstatic = true;
            } else {
                System.out.println("Unsupported: "+type);
                return false;
            }
        } else {
            if (type.equals("InstanceMember")) {
                isglobal = false;
                isstatic = false;
            } else if (type.equals("StaticMember")) {
                isglobal = false;
                isstatic = true;
            } else {
                System.out.println("Unsupported: "+type);
                return false;
            }
        }
        
        SSAClass clazz;
        if (isglobal) clazz = globalClass;
        else {
            int classNum = getIndex(line, colon2+2, "Class");
            clazz = (SSAClass)classes.get(classNum);
        }
        
        if (isProc) {
            SSAMethod meth = new SSAMethod(Utf8.get(name), fileLocation, clazz, isstatic,
					   isConstructor, isCompilerGenerated);
            setArrayListElem(methods, number, meth);
        } else {
            SSAField field = new SSAField(Utf8.get(name), fileLocation, clazz, isstatic);
            setArrayListElem(fields, number, field);
        }
        
        return true;
    }

    boolean readDispatch(String line) {
        if (!line.startsWith("DISPATCH")) return false;
        
        int open1 = line.indexOf('[');
        int open2 = line.indexOf('[',open1+1);
        int close1 = line.indexOf(']');
        int close2 = line.indexOf(']',close1+1);
        
        int targetProcNum = Integer.parseInt(line.substring(open1+1, close1));
        int baseProcNum = Integer.parseInt(line.substring(open2+1, close2));

        ((SSAMethod)methods.get(baseProcNum)).addCallTarget((SSAMethod)methods.get(targetProcNum));

        return true;
    }

    SSAType readType(String properties) {
	int pointerDepth = 0;
	int classNum = -1;
	boolean reference = false;

	StringTokenizer st = new StringTokenizer(properties, " ,");
	
	while (st.hasMoreTokens()) {
	    String token = st.nextToken();
	    if (token.equals("reference")) reference = true;
	    else if (token.startsWith("Class")) {
		classNum = Integer.parseInt(token.substring(token.indexOf('<')+1, token.lastIndexOf('>')));
	    } else if (token.startsWith("depth")) {
		pointerDepth = Integer.parseInt(token.substring(token.indexOf('<')+1, token.lastIndexOf('>')));
	    }
	}

	if (classNum == -1) {
	    // we aren't given adequate type information - use universal type
	    return universalType;
	}

	SSAClass clazz = (SSAClass)classes.get(classNum);
	SSAType type = clazz.getType().getPointerToThis(pointerDepth);
	if (reference) return type.getReferenceToThis();
	return type;
    }
    
    static boolean assignBaseToNode(Node base, Node dest) {
        if (base.hasEdge(null, dest)) return false;
        base.addEdge(null,dest,null);
        return true;
    }
    
    static boolean assignBaseToBase(Node base, Node dest) {
        LinkedHashSet edges = new LinkedHashSet();
        dest.getEdges(null, edges);
        boolean changed = false;
        for (Iterator i = edges.iterator(); i.hasNext();) {
            if (assignBaseToNode(base, (Node)i.next())) 
                changed = true;
        }
        return changed;
    }
    
    void readProcdef(String line1, BufferedReader input) throws IOException {
        int methodNum = Integer.parseInt(line1.substring(line1.indexOf('[')+1, line1.indexOf(']')));
        SSAMethod method = (SSAMethod)methods.get(methodNum);
        
        ArrayList files = new ArrayList();
        ArrayList fileLocations = new ArrayList();
        ArrayList localVars = new ArrayList();
        ArrayList vvNodes = new ArrayList();
        
        ArrayList paramNodes = new ArrayList();
        ArrayList returnVVNodeNums = new ArrayList();
        ArrayList methodCalls = new ArrayList();

        HashMap callToRVN = new HashMap();
        HashMap callToTEN = new HashMap();
        LinkedHashSet passedAsParameter = new LinkedHashSet();
        
        ArrayList instructions = new ArrayList();
        
        for (;;) {
            String line = input.readLine().trim();
            if (line.length()==0) continue;
            if (line.startsWith("END ProcDef")) break;
            
            if (line.startsWith("File")) {
                int fileNum = getIndex(line, 0, "File");
                setArrayListElem(files, fileNum, line.substring(line.indexOf('"')));
            } else if (line.startsWith("LocMarker")) {
                int locNum = getIndex(line, 0, "LocMarker");
                int fileNum = getIndex(line, line.indexOf(':')+2, "File");
                String lineNum = line.substring(line.lastIndexOf(' ')+1);
                setArrayListElem(fileLocations, locNum, ((String)files.get(fileNum)+':'+lineNum).intern());
            } else if (line.startsWith("LocalVar")) {
                int varNum = getIndex(line, 0, "LocalVar");
                String name = line.substring(line.indexOf('"')+1, line.lastIndexOf('"'));
		SSAType type = readType(line.substring(line.indexOf('[')+1, line.lastIndexOf(']')));
                setArrayListElem(localVars, varNum, new LocalVar(Utf8.get(name), type));
            } else if (line.startsWith("VVNode")) {
                int vvNum = getIndex(line, 0, "VVNode");
                int varPos = line.indexOf(':')+2;
                if (!line.startsWith("LocalVar", varPos)) continue; // ignore members etc.
                int varNum = getIndex(line, varPos, "LocalVar");
                LocalVar localVar = (LocalVar)localVars.get(varNum);
                VVNode vvNode = new VVNode(localVar);
                localVar.addVVNode(vvNode);
                setArrayListElem(vvNodes, vvNum, vvNode);
            } else if (line.startsWith("PHI") || line.startsWith("LAMBDA")) {
                instructions.add(line);
            } else if (line.startsWith("CALL_TARGET")) {
                StringTokenizer st = new StringTokenizer(line.substring(line.indexOf('(')+1, line.length()-1), ", ");
                int callSiteNum = getIndex(st.nextToken(), 0, "CallSite");
                int calledMethodNum = getIndex(st.nextToken(), 0, "Proc");
                
                SSAMethod calledMethod = (SSAMethod)methods.get(calledMethodNum);
                
                ProgramLocation mc = new ProgramLocation.SSAProgramLocation(programLocationCounter++,
                                                                            method, calledMethod);
                setArrayListElem(methodCalls, callSiteNum, mc);
            } else if (line.startsWith("Constraint: ")) {
                String constraint = line.substring("Constraint: ".length());
                
                if (constraint.startsWith("PROC_PARAM_ARG")) {
                    StringTokenizer st = new StringTokenizer(constraint.substring(constraint.indexOf('(')), "(), ");
                    st.nextToken(); // just tells us the proc - we already know it
                    String argPos = st.nextToken();
                    int vvNum = getIndex(st.nextToken(), 0, "VVNode");

                    if (argPos.startsWith("arg_in")) {
                        VVNode vvNode = (VVNode)vvNodes.get(vvNum);
                        LocalVar localVar = vvNode.getLocalVar();

                        int argNum = getIndex(argPos, 0, "arg_in");
                        ParamNode paramNode = new ParamNode(method, argNum, localVar.getType());
                        
                        setArrayListElem(paramNodes, argNum, paramNode);
                        assignBaseToNode(vvNode.getNode(), paramNode);
                    } else {
                        returnVVNodeNums.add(new Integer(vvNum));
                    }
                } else if (constraint.startsWith("ALLOC")) { // new
                    int vvNum = getIndex(constraint, constraint.indexOf('(')+1, "VVNode");
                    VVNode vvNode = (VVNode)vvNodes.get(vvNum);
                    LocalVar localVar = vvNode.getLocalVar();

                    assignBaseToNode(vvNode.getNode(), new ConcreteTypeNode(localVar.getType()));
                } else if (constraint.startsWith("ASSIGN") 
                            || constraint.startsWith("SELF_ASSIGN")
                            || constraint.startsWith("ARG_ASSIGN")
                            || constraint.startsWith("CALL_PARAM_ARG")) {
                    instructions.add(constraint);
                }
            }
        }
        
        boolean changed = true;
        while (changed) {
            changed = false;
            
            for (int i=0; i<instructions.size(); i++) {
                String inst = (String)instructions.get(i);

                if (inst.startsWith("ASSIGN")) {
                    StringTokenizer st = new StringTokenizer(inst.substring(inst.indexOf('(')), "(), ");
                    int inVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode inVV = (VVNode)vvNodes.get(inVVNum);
                    int destVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode destVV = (VVNode)vvNodes.get(destVVNum);
                    int outVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode outVV = (VVNode)vvNodes.get(outVVNum);

                    if (assignBaseToBase(destVV.getNode(), inVV.getNode())) changed = true;
                    if (assignBaseToBase(outVV.getNode(), inVV.getNode())) changed = true;
                } else if (inst.startsWith("SELF_ASSIGN") || inst.startsWith("ARG_ASSIGN")) {
                    StringTokenizer st = new StringTokenizer(inst.substring(inst.indexOf('(')), "(), ");
                    int inVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode inVV = (VVNode)vvNodes.get(inVVNum);
                    int destVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode destVV = (VVNode)vvNodes.get(destVVNum);

                    if (assignBaseToBase(destVV.getNode(), inVV.getNode())) changed = true;
                } else if (inst.startsWith("LAMBDA")) {
                    StringTokenizer st = new StringTokenizer(inst.substring(inst.indexOf('(')+1, inst.length()-1), ", ");
                    int inVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode inVV = (VVNode)vvNodes.get(inVVNum);
                    while (st.hasMoreTokens()) {
                        String token=st.nextToken();
                        if (token.startsWith("VVNode")) {
                            int destVVNum = getIndex(token, 0, "VVNode");
                            VVNode destVV = (VVNode)vvNodes.get(destVVNum);

                            if (assignBaseToBase(destVV.getNode(), inVV.getNode())) changed = true;
                        }
                    }
                } else if (inst.startsWith("PHI")) {
                    StringTokenizer st = new StringTokenizer(inst.substring(inst.indexOf('(')+1, inst.length()-1), ", ");
                    int destVVNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode destVV = (VVNode)vvNodes.get(destVVNum);
                    while (st.hasMoreTokens()) {
                        String token=st.nextToken();
                        if (token.startsWith("VVNode")) {
                            int inVVNum = getIndex(token, 0, "VVNode");
                            VVNode inVV = (VVNode)vvNodes.get(inVVNum);

                            if (assignBaseToBase(destVV.getNode(), inVV.getNode())) changed = true;
                        }
                    }
                } else if (inst.startsWith("CALL_PARAM_ARG")) {
                    StringTokenizer st = new StringTokenizer(inst.substring(inst.indexOf('(')+1, inst.length()-1), ", ");
                    int callSiteNum = getIndex(st.nextToken(), 0, "CallSite");
                    ProgramLocation mc = (ProgramLocation)methodCalls.get(callSiteNum);
                    String argStr = st.nextToken();
                    int vvNum = getIndex(st.nextToken(), 0, "VVNode");
                    VVNode vvNode = (VVNode)vvNodes.get(vvNum);

                    if (argStr.startsWith("return")) {
                        ReturnValueNode n = (ReturnValueNode)callToRVN.get(mc);
                        if (n == null) {
                            callToRVN.put(mc, n = new ReturnValueNode(mc));
                            passedAsParameter.add(n);
                        }
                        if (assignBaseToNode(vvNode.getNode(), n)) changed = true;
                    } else {
                        int argIdx = getIndex(argStr, 0, "arg_in");

                        LinkedHashSet sourceNodes = new LinkedHashSet();
                        vvNode.getNode().getEdges(null, sourceNodes);

                        for (Iterator it = sourceNodes.iterator(); it.hasNext(); ) {
                            Node n = (Node)it.next();
                            if (n.recordPassedParameter(mc, argIdx)) changed = true;
                            if (passedAsParameter.add(n)) changed = true;
                        }
                    }
                } else {
                    System.out.println("Unexpected instruction: "+inst);
                    throw new IllegalArgumentException(); // 
                }
            }
        }
        
        ParamNode[] paramNodeArray = new ParamNode[paramNodes.size()];
        for (int i=0; i<paramNodes.size(); i++) 
            paramNodeArray[i]=(ParamNode)paramNodes.get(i);
        
        GlobalNode myGlobal = new GlobalNode();
        LinkedHashSet methodCallsSet = new LinkedHashSet(methodCalls);
        Set returned = new LinkedHashSet();
        for (int i=0; i<returnVVNodeNums.size(); i++) {
            int vvNum = ((Integer)returnVVNodeNums.get(i)).intValue();
            Node baseNode = ((VVNode)vvNodes.get(vvNum)).getNode();
            baseNode.getEdges(null, returned);
        }
        
        LinkedHashSet thrown = new LinkedHashSet();

	if (method.isConstructor) {
	    // force return to be concreteTypeNode so we know the datatype
	    SSAType returnType = method.getDeclaringClass().getType().getPointerToThis();

	    returned = Collections.singleton(new ConcreteTypeNode(returnType));
	}
        
        MethodSummary methodSummary = new MethodSummary(method, paramNodeArray, myGlobal, methodCallsSet, callToRVN, callToTEN, returned, thrown, passedAsParameter);
        method.setSummary(methodSummary);

        SSAType[] paramTypes = new SSAType[paramNodes.size()];
        for (int i=0; i<paramNodes.size(); i++) {
            paramTypes[i] = (SSAType)((ParamNode)paramNodes.get(i)).getDeclaredType();
        }
        method.setParamTypes(paramTypes);
        
	//	System.out.println(methodSummary);
    }
    
    /** Creates a new instance of SSAReader */
    public SSAReader(InputStream inputStream) {
        BufferedReader input = new BufferedReader(new InputStreamReader(inputStream));

        classes = new ArrayList();
        fields = new ArrayList();
        methods = new ArrayList();

        globalClass = new SSAClass(Utf8.get("___global_"), "");
        
        programLocationCounter = 0;
        
        try {
            for (;;) {
                String line = input.readLine();
                if (line == null) break; // EOF
                line = line.trim();
                
                if (line.length()==0) continue; // skip blanks
                
                if (line.startsWith("BEGIN ProcDef")) {
                    readProcdef(line, input);
                    continue;
                }
                
                if (readClass(line)) continue;
                if (readSuperclass(line)) continue;
                if (readMember(line)) continue;
                if (readDispatch(line)) continue;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        // precompute class ancestors for fast instanceOf computation
        for (int i=0; i<classes.size(); i++) {
            ((SSAClass)classes.get(i)).computeAncestors();
        }
    }
    
    // should be called once after the constructor has read the SSA file, 
    // and this object should then be discarded
    public ArrayList getMethods() { return methods; }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        HostedVM.initialize();
        
        int index = 0; int index0 = -1;
        while (index != index0 && index < args.length) {
            index = Main.TraceFlags.setTraceFlag(args, index0 = index);
        }
        
        try {

            InputStream istream = System.in;
            if (args.length>index) {
                String filename = args[index];
                istream = new FileInputStream(filename);
                if (filename.endsWith(".gz"))
                   istream = new GZIPInputStream(istream);
            }
        
            ArrayList methods;
            {
                SSAReader reader = new SSAReader(istream);
                methods = reader.getMethods();
            } // allow reader to pass out of scope so the garbage collector can clean up

            AndersenPointerAnalysis apa = new AndersenPointerAnalysis(false);
            for (int i=0; i<methods.size(); i++) {
                MethodSummary summary = ((SSAMethod)methods.get(i)).getSummary();
                if (summary!=null)
                    apa.addToRootSet(summary);
            }
            apa.iterate();
            
            System.out.println(apa.getCallGraph());
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}
