/*
 * TraceFlags.java
 *
 * Created on February 1, 2001, 12:35 PM
 *
 * @author  John Whaley
 * @version 
 */

package Main;

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
            Interpreter.Interpreter.ALWAYS_TRACE = true;
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
        return i;
    }
}
