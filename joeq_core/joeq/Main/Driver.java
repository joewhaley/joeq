/*
 * Driver.java
 *
 * Created on January 9, 2002, 9:17 AM
 *
 * @author  John Whaley
 * @version 
 */

package Main;

import Clazz.*;
import Compil3r.Quad.*;
import java.io.*;
import java.util.*;
import jq;
import Run_Time.*;
import Bootstrap.*;
import UTF.*;

public abstract class Driver {

    public static void main(String[] args) {
        // initialize jq
        jq.initializeForHostJVMExecution();
        
        try {
            interpreterClass = Class.forName("Interpreter.QuadInterpreter$State",
                                             false, Driver.class.getClassLoader());
        } catch (ClassNotFoundException x) {
            System.err.println("Warning: interpreter class not found.");
        }
        
        if ((args.length == 0) || args[0].equals("-i")) {
            // interactive mode
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            for (;;) {
                String[] commands;
                try {
                    System.out.print("joeq> ");
                    String line = in.readLine();
                    if (line == null) return;
                    StringTokenizer st = new StringTokenizer(line);
                    int size = st.countTokens();
                    commands = new String[size];
                    for (int i=0; i<size; ++i) {
                        commands[i] = st.nextToken();
                    }
                    jq.assert(!st.hasMoreTokens());
                } catch (IOException x) {
                    System.err.println(x.toString());
                    return;
                }
                for (int i=0; i<commands.length; ++i) {
                    i = processCommand(commands, i);
                }
            }
        }
        for (int i=0; i<args.length; ++i) {
            i = processCommand(args, i);
        }
    }

    static List classesToProcess = new LinkedList();
    static HashSet methodNamesToProcess;
    static boolean trace_bb = false;
    static boolean trace_cfg = false;
    static boolean trace_method = false;
    static boolean trace_type = false;
    
    static Class interpreterClass;
    
    public static int processCommand(String[] commandBuffer, int index) {
        try {
            if (commandBuffer[index].equalsIgnoreCase("addtoclasspath")) {
                String path = commandBuffer[++index];
                PrimordialClassLoader.loader.addToClasspath(path);
            } else if (commandBuffer[index].equalsIgnoreCase("trace")) {
                String which = commandBuffer[++index];
                if (which.equalsIgnoreCase("bb")) {
                    trace_bb = true;
                } else if (which.equalsIgnoreCase("cfg")) {
                    trace_cfg = true;
                } else if (which.equalsIgnoreCase("method")) {
                    trace_method = true;
                } else if (which.equalsIgnoreCase("type")) {
                    trace_type = true;
                } else {
                    System.err.println("Unknown trace option "+which);
                }
            } else if (commandBuffer[index].equalsIgnoreCase("method")) {
                if (methodNamesToProcess == null) methodNamesToProcess = new HashSet();
                methodNamesToProcess.add(commandBuffer[++index]);
            } else if (commandBuffer[index].equalsIgnoreCase("class")) {
                String canonicalClassName = canonicalizeClassName(commandBuffer[++index]);
                try {
                    jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(canonicalClassName);
                    c.load();
                    classesToProcess.add(c);
                } catch (NoClassDefFoundError x) {
                    System.err.println("Class "+commandBuffer[index]+" (canonical name "+canonicalClassName+") not found.");
                }
            } else if (commandBuffer[index].equalsIgnoreCase("package")) {
                String canonicalPackageName = commandBuffer[++index].replace('.','/');
                if (!canonicalPackageName.endsWith("/")) canonicalPackageName += '/';
                Iterator i = PrimordialClassLoader.loader.listPackage(canonicalPackageName);
                if (!i.hasNext()) {
                    System.err.println("Package "+canonicalPackageName+" not found.");
                }
                while (i.hasNext()) {
                    String canonicalClassName = canonicalizeClassName((String)i.next());
                    try {
                        jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(canonicalClassName);
                        c.load();
                        classesToProcess.add(c);
                    } catch (NoClassDefFoundError x) {
                        System.err.println("Class "+commandBuffer[index]+" (canonical name "+canonicalClassName+") not found.");
                    }
                }
            } else if (commandBuffer[index].equalsIgnoreCase("setinterpreter")) {
                String interpreterClassName = commandBuffer[++index];
                try {
                    Class cl = Class.forName(interpreterClassName);
                    if (Class.forName("Interpreter.QuadInterpreter$State").isAssignableFrom(cl)) {
                        interpreterClass = cl;
			System.out.println("Interpreter class changed to "+interpreterClass);
                    } else {
                        System.err.println("Class "+interpreterClassName+" does not subclass Interpreter.QuadInterpreter.State.");
                    }
                } catch (java.lang.ClassNotFoundException x) {
                    System.err.println("Cannot find interpreter named "+interpreterClassName+".");
                    System.err.println("Check your classpath and make sure you compiled your interpreter.");
                    return index;
                }
                
            } else if (commandBuffer[index].equalsIgnoreCase("interpret")) {
                String fullName = commandBuffer[++index];
                int b = fullName.lastIndexOf('.')+1;
                String methodName = fullName.substring(b);
                String className = canonicalizeClassName(fullName.substring(0, b-1));
                try {
                    jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(className);
                    c.load(); c.verify(); c.prepare(); c.sf_initialize(); c.cls_initialize();
                    jq_StaticMethod m = null;
                    Utf8 rootm_name = Utf8.get(methodName);
                    for(Iterator it = java.util.Arrays.asList(c.getDeclaredStaticMethods()).iterator();
                        it.hasNext(); ) {
                        jq_StaticMethod sm = (jq_StaticMethod)it.next();
                        if (sm.getName() == rootm_name) {
                            m = sm;
                            break;
                        }
                    }
                    if (m != null) {
                        Object[] args = new Object[m.getParamTypes().length];
                        index = parseMethodArgs(args, m.getParamTypes(), commandBuffer, index);
                        Interpreter.QuadInterpreter.State s = null;
                        java.lang.reflect.Method im = interpreterClass.getMethod("interpretMethod", new Class[] { Class.forName("Clazz.jq_Method"), new Object[0].getClass()});
                        s = (Interpreter.QuadInterpreter.State)im.invoke(null, new Object[] {m, args});
                        //s = Interpreter.QuadInterpreter.State.interpretMethod(m, args);
                        System.out.flush();
                        System.out.println("Result of interpretation: "+s);
                    } else {
                        System.err.println("Class "+fullName.substring(0, b-1)+" doesn't contain a void static no-argument method with name "+methodName);
                    }
                } catch (NoClassDefFoundError x) {
                    System.err.println("Class "+fullName.substring(0, b-1)+" (canonical name "+className+") not found.");
                    return index;
                } catch (NoSuchMethodException x) {
                    System.err.println("Interpreter method in "+interpreterClass+" not found! "+x);
                    return index;
                } catch (ClassNotFoundException x) {
                    System.err.println("Clazz.jq_Method class not found! "+x);
                    return index;
                } catch (IllegalAccessException x) {
                    System.err.println("Cannot access interpreter "+interpreterClass+": "+x);
                    return index;
                } catch (java.lang.reflect.InvocationTargetException x) {
                    System.err.println("Interpreter threw exception: "+x.getTargetException());
                    x.getTargetException().printStackTrace();
                    return index;
                }
                
            } else if (commandBuffer[index].equalsIgnoreCase("set")) {
                String fullName = commandBuffer[++index];
                int b = fullName.lastIndexOf('.')+1;
                String fieldName = fullName.substring(b);
                String className = canonicalizeClassName(fullName.substring(0, b-1));
                try {
                    jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(className);
                    c.load(); c.verify(); c.prepare(); c.sf_initialize(); c.cls_initialize();
                    jq_StaticField m = null;
                    Utf8 sf_name = Utf8.get(fieldName);
                    for(Iterator it = java.util.Arrays.asList(c.getDeclaredStaticFields()).iterator();
                        it.hasNext(); ) {
                        jq_StaticField sm = (jq_StaticField)it.next();
                        if (sm.getName() == sf_name) {
                            m = sm;
                            break;
                        }
                    }
                    if (m != null) {
                        java.lang.reflect.Field f = (java.lang.reflect.Field)Reflection.getJDKMember(m);
                        f.setAccessible(true);
                        Object[] o = new Object[1];
                        index = parseArg(o, 0, m.getType(), commandBuffer, index);
                        f.set(null, o[0]);
                    } else {
                        System.err.println("Class "+fullName.substring(0, b-1)+" doesn't contain a static field with name "+fieldName);
                    }
                } catch (NoClassDefFoundError x) {
                    System.err.println("Class "+fullName.substring(0, b-1)+" (canonical name "+className+") not found.");
                    return index;
                } catch (IllegalAccessException x) {
                    System.err.println("Cannot access field: "+x);
                    return index;
                }
                
            } else if (commandBuffer[index].equalsIgnoreCase("addpass")) {
                String passname = commandBuffer[++index];
                ControlFlowGraphVisitor mv = null; BasicBlockVisitor bbv = null; QuadVisitor qv = null;
                Object o;
                try {
                    Class c = Class.forName(passname);
                    o = c.newInstance();
                    if (o instanceof ControlFlowGraphVisitor) {
                        mv = (ControlFlowGraphVisitor)o;
                    } else {
                        if (o instanceof BasicBlockVisitor) {
                            bbv = (BasicBlockVisitor)o;
                        } else {
                            if (o instanceof QuadVisitor) {
                                qv = (QuadVisitor)o;
                            } else {
                                System.err.println("Unknown pass type "+c);
                                return index;
                            }
                            bbv = new QuadVisitor.AllQuadVisitor(qv, trace_bb);
                        }
                        mv = new BasicBlockVisitor.AllBasicBlockVisitor(bbv, trace_method);
                    }
                    CodeCache.passes.add(mv);
                } catch (java.lang.ClassNotFoundException x) {
                    System.err.println("Cannot find pass named "+passname+".");
                    System.err.println("Check your classpath and make sure you compiled your pass.");
                    return index;
                } catch (java.lang.InstantiationException x) {
                    System.err.println("Cannot instantiate pass "+passname+": "+x);
                    return index;
                } catch (java.lang.IllegalAccessException x) {
                    System.err.println("Cannot access pass "+passname+": "+x);
                    System.err.println("Be sure that you made your class public?");
                    return index;
                }
            } else if (commandBuffer[index].equalsIgnoreCase("runpass")) {
                String passname = commandBuffer[++index];
                jq_TypeVisitor cv = null; jq_MethodVisitor mv = null; ControlFlowGraphVisitor cfgv = null; BasicBlockVisitor bbv = null; QuadVisitor qv = null;
                Object o;
                try {
                    Class c = Class.forName(passname);
                    o = c.newInstance();
                    if (o instanceof jq_TypeVisitor) {
                        cv = (jq_TypeVisitor)o;
                    } else {
                        if (o instanceof jq_MethodVisitor) {
                            mv = (jq_MethodVisitor)o;
                        } else {
                            if (o instanceof ControlFlowGraphVisitor) {
                                cfgv = (ControlFlowGraphVisitor)o;
                            } else {
                                if (o instanceof BasicBlockVisitor) {
                                    bbv = (BasicBlockVisitor)o;
                                } else {
                                    if (o instanceof QuadVisitor) {
                                        qv = (QuadVisitor)o;
                                    } else {
                                        System.err.println("Unknown pass type "+c);
                                        return index;
                                    }
                                    bbv = new QuadVisitor.AllQuadVisitor(qv, trace_bb);
                                }
                                cfgv = new BasicBlockVisitor.AllBasicBlockVisitor(bbv, trace_cfg);
                            }
                            mv = new ControlFlowGraphVisitor.CodeCacheVisitor(cfgv, trace_method);
                        }
                        cv = new jq_MethodVisitor.DeclaredMethodVisitor(mv, methodNamesToProcess, trace_type);
                    }
                } catch (java.lang.ClassNotFoundException x) {
                    System.err.println("Cannot find pass named "+passname+".");
                    System.err.println("Check your classpath and make sure you compiled your pass.");
                    return index;
                } catch (java.lang.InstantiationException x) {
                    System.err.println("Cannot instantiate pass "+passname+": "+x);
                    return index;
                } catch (java.lang.IllegalAccessException x) {
                    System.err.println("Cannot access pass "+passname+": "+x);
                    System.err.println("Be sure that you made your class public?");
                    return index;
                }
                for (Iterator i = classesToProcess.iterator(); i.hasNext(); ) {
                    jq_Type t = (jq_Type)i.next();
                    try {
                        t.accept(cv);
                    } catch (Exception x) {
                        System.err.println("Runtime exception occurred while executing pass on "+t+" : "+x);
                        x.printStackTrace(System.err);
                    }
                }
                System.out.println("Completed pass! "+o);
            } else if (commandBuffer[index].equalsIgnoreCase("exit") || commandBuffer[index].equalsIgnoreCase("quit")) {
                System.exit(0);
            } else {
		int index2 = TraceFlags.setTraceFlag(commandBuffer, index);
		if (index == index2)
		    System.err.println("Unknown command "+commandBuffer[index]);
                else
                    index = index2-1;
            }
        } catch (ArrayIndexOutOfBoundsException x) {
            System.err.println("Incomplete command");
        }
        return index;
    }
    public static String canonicalizeClassName(String s) {
        if (s.endsWith(".class")) s = s.substring(0, s.length()-6);
        s = s.replace('.', '/');
        String desc = "L"+s+";";
        return desc;
    }
    
    public static int parseArg(Object[] args, int m, jq_Type type, String[] s_args, int j) {
        if (type == PrimordialClassLoader.loader.getJavaLangString())
            args[m] = s_args[++j];
        else if (type == jq_Primitive.BOOLEAN)
            args[m] = Boolean.valueOf(s_args[++j]);
        else if (type == jq_Primitive.BYTE)
            args[m] = Byte.valueOf(s_args[++j]);
        else if (type == jq_Primitive.SHORT)
            args[m] = Short.valueOf(s_args[++j]);
        else if (type == jq_Primitive.CHAR)
            args[m] = new Character(s_args[++j].charAt(0));
        else if (type == jq_Primitive.INT)
            args[m] = Integer.valueOf(s_args[++j]);
        else if (type == jq_Primitive.LONG) {
            args[m] = Long.valueOf(s_args[++j]);
        } else if (type == jq_Primitive.FLOAT)
            args[m] = Float.valueOf(s_args[++j]);
        else if (type == jq_Primitive.DOUBLE) {
            args[m] = Double.valueOf(s_args[++j]);
        } else if (type.isArrayType()) {
            if (!s_args[++j].equals("{")) 
                jq.UNREACHABLE("array parameter doesn't start with {");
            int count=0;
            while (!s_args[++j].equals("}")) ++count;
            jq_Type elementType = ((jq_Array)type).getElementType();
            if (elementType == PrimordialClassLoader.loader.getJavaLangString()) {
                String[] array = new String[count];
                for (int k=0; k<count; ++k)
                    array[k] = s_args[j-count+k];
                args[m] = array;
            } else if (elementType == jq_Primitive.BOOLEAN) {
                boolean[] array = new boolean[count];
                for (int k=0; k<count; ++k)
                    array[k] = Boolean.valueOf(s_args[j-count+k]).booleanValue();
                args[m] = array;
            } else if (elementType == jq_Primitive.BYTE) {
                byte[] array = new byte[count];
                for (int k=0; k<count; ++k)
                    array[k] = Byte.parseByte(s_args[j-count+k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.SHORT) {
                short[] array = new short[count];
                for (int k=0; k<count; ++k)
                    array[k] = Short.parseShort(s_args[j-count+k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.CHAR) {
                char[] array = new char[count];
                for (int k=0; k<count; ++k)
                    array[k] = s_args[j-count+k].charAt(0);
                args[m] = array;
            } else if (elementType == jq_Primitive.INT) {
                int[] array = new int[count];
                for (int k=0; k<count; ++k)
                    array[k] = Integer.parseInt(s_args[j-count+k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.LONG) {
                long[] array = new long[count];
                for (int k=0; k<count; ++k)
                    array[k] = Long.parseLong(s_args[j-count+k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.FLOAT) {
                float[] array = new float[count];
                for (int k=0; k<count; ++k)
                    array[k] = Float.parseFloat(s_args[j-count+k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.DOUBLE) {
                double[] array = new double[count];
                for (int k=0; k<count; ++k)
                    array[k] = Double.parseDouble(s_args[j-count+k]);
                args[m] = array;
            } else
                jq.UNREACHABLE("Parsing of type "+type+" is not implemented");
        } else
            jq.UNREACHABLE("Parsing of type "+type+" is not implemented");
        return j;
    }
    
    public static int parseMethodArgs(Object[] args, jq_Type[] paramTypes, String[] s_args, int j) {
        try {
            for (int i=0, m=0; i<paramTypes.length; ++i, ++m) {
                j = parseArg(args, m, paramTypes[i], s_args, j);
            }
        } catch (ArrayIndexOutOfBoundsException x) {
            x.printStackTrace();
            jq.UNREACHABLE("not enough method arguments");
        }
        return j;
    }
}
