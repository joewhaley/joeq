/*
 * TraceFlags.java
 *
 * Created on February 1, 2001, 12:35 PM
 *
 * @author  John Whaley
 * @version 
 */

package Main;

import Clazz.*;
import UTF.Utf8;
import Bootstrap.PrimordialClassLoader;
import Run_Time.Reflection;
import java.util.Iterator;

public abstract class TraceFlags {

    public static int setTraceFlag(String[] args, int i) {
        if (args[i].equalsIgnoreCase("-TraceCodeAllocator")) {
            Allocator.CodeAllocator.TRACE = true;
            Allocator.RuntimeCodeAllocator.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceAssembler")) {
            Assembler.x86.x86Assembler.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceBC2Quad")) {
            Compil3r.Quad.BytecodeToQuad.ALWAYS_TRACE = true;
            Compil3r.Quad.BytecodeToQuad.AbstractState.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceLiveRef")) {
            Compil3r.BytecodeAnalysis.LiveRefAnalysis.ALWAYS_TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceBootImage")) {
            Bootstrap.BootImage.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceObjectTraverser")) {
            Bootstrap.ObjectTraverser.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceClassLoader")) {
            Bootstrap.PrimordialClassLoader.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceClass")) {
            Clazz.jq_Class.TRACE = true;
            Clazz.jq_Array.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceExceptions")) {
            Clazz.jq_CompiledCode.TRACE = true;
            Compil3r.Reference.x86.x86ReferenceExceptionDeliverer.TRACE = true;
            Run_Time.ExceptionDeliverer.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceTrimmer")) {
            Compil3r.BytecodeAnalysis.Trimmer.TRACE = true;
            Bootstrap.BootstrapRootSet.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceCompiler")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.ALWAYS_TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceCompileStubs")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.TRACE_STUBS = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceLinker")) {
            Compil3r.Reference.x86.x86ReferenceLinker.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceInterpreter")) {
            Interpreter.BytecodeInterpreter.ALWAYS_TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceQuadInterpreter")) {
            Interpreter.QuadInterpreter.State.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceStackWalker")) {
            Run_Time.StackWalker.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceUtf8")) {
            UTF.Utf8.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceScheduler")) {
            Scheduler.jq_NativeThread.TRACE = true;
            Scheduler.jq_InterrupterThread.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceLocks")) {
            Run_Time.Monitor.TRACE = true;
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceByMethodName")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.TraceMethod_MethodNames.add(args[++i]);
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceByClassName")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.TraceMethod_ClassNames.add(args[++i]);
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceBCByMethodName")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.TraceBytecode_MethodNames.add(args[++i]);
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-TraceBCByClassName")) {
            Compil3r.Reference.x86.x86ReferenceCompiler.TraceBytecode_ClassNames.add(args[++i]);
            return i+1;
        }
        if (args[i].equalsIgnoreCase("-Set")) {
            String fullName = args[++i];
            int b = fullName.lastIndexOf('.') + 1;
            String fieldName = fullName.substring(b);
            String className = fullName.substring(0, b - 1);
            try {
                jq_Class c = (jq_Class) jq.parseType(className);
                c.load();
                c.verify();
                c.prepare();
                c.sf_initialize();
                c.cls_initialize();
                jq_StaticField m = null;
                Utf8 sf_name = Utf8.get(fieldName);
                for (Iterator it = java.util.Arrays.asList(c.getDeclaredStaticFields()).iterator(); it.hasNext();) {
                    jq_StaticField sm = (jq_StaticField) it.next();
                    if (sm.getName() == sf_name) {
                        m = sm;
                        break;
                    }
                }
                if (m != null) {
                    java.lang.reflect.Field f = (java.lang.reflect.Field) Reflection.getJDKMember(m);
                    f.setAccessible(true);
                    Object[] o = new Object[1];
                    i = parseArg(o, 0, m.getType(), args, i);
                    f.set(null, o[0]);
                } else {
                    System.err.println("Class " + fullName.substring(0, b - 1) + " doesn't contain a static field with name " + fieldName);
                }
            } catch (NoClassDefFoundError x) {
                System.err.println("Class " + fullName.substring(0, b - 1) + " (canonical name " + className + ") not found.");
                return i;
            } catch (IllegalAccessException x) {
                System.err.println("Cannot access field: " + x);
                return i+1;
            }
            return i+1;
        }
        return i;
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
            int count = 0;
            while (!s_args[++j].equals("}")) ++count;
            jq_Type elementType = ((jq_Array) type).getElementType();
            if (elementType == PrimordialClassLoader.loader.getJavaLangString()) {
                String[] array = new String[count];
                for (int k = 0; k < count; ++k)
                    array[k] = s_args[j - count + k];
                args[m] = array;
            } else if (elementType == jq_Primitive.BOOLEAN) {
                boolean[] array = new boolean[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Boolean.valueOf(s_args[j - count + k]).booleanValue();
                args[m] = array;
            } else if (elementType == jq_Primitive.BYTE) {
                byte[] array = new byte[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Byte.parseByte(s_args[j - count + k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.SHORT) {
                short[] array = new short[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Short.parseShort(s_args[j - count + k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.CHAR) {
                char[] array = new char[count];
                for (int k = 0; k < count; ++k)
                    array[k] = s_args[j - count + k].charAt(0);
                args[m] = array;
            } else if (elementType == jq_Primitive.INT) {
                int[] array = new int[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Integer.parseInt(s_args[j - count + k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.LONG) {
                long[] array = new long[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Long.parseLong(s_args[j - count + k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.FLOAT) {
                float[] array = new float[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Float.parseFloat(s_args[j - count + k]);
                args[m] = array;
            } else if (elementType == jq_Primitive.DOUBLE) {
                double[] array = new double[count];
                for (int k = 0; k < count; ++k)
                    array[k] = Double.parseDouble(s_args[j - count + k]);
                args[m] = array;
            } else
                jq.UNREACHABLE("Parsing of type " + type + " is not implemented");
        } else
            jq.UNREACHABLE("Parsing of type " + type + " is not implemented");
        return j;
    }
    
}
