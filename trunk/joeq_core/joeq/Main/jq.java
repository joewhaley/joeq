/*
 * jq.java
 *
 * Created on December 19, 2000, 11:15 AM
 *
 */
package Main;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import Bootstrap.MethodInvocation;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_NameAndDesc;
import Clazz.jq_Primitive;
import Clazz.jq_Reference;
import Clazz.jq_StaticMethod;
import Clazz.jq_Type;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.Reflection;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_MainThread;
import Scheduler.jq_NativeThread;
import Scheduler.jq_Thread;
import UTF.Utf8;

/**
 *
 * @author  John Whaley
 * @version $Id$
 */

// FIXED BUGS:
//  off-by-one error jq_Atom
//  alg error constantpool forward refs
//  cut & paste error, used old idx rather than new one
//  misunderstanding of class file descriptor spec
//  when initializing type descriptors, byte to string converts to int, not char
//  cut & paste error, forgot to change memcmp in utf8.get
//  forgot a "break" in big bytecode traverser switch statement
//  off-by-one error, index should start at -1 in bytecode traverser
//  parsed array descriptors in constant pool as class descriptors
//  forgot to set constant_pool_tag to resolved when field is known at load time
//  forgot to set resolved flag in InstanceField.resolve()
//  java.lang.Character contains invalid UTF8 (-32,-108,-124), disabled strict checking.
//  forgot to add root set of classes to Trimmer, so codegen of new jq_Class failed
//  added default types to Trimmer without preparing them
//  forgot to recompile Trimmer after change
//  assumed entrySet() returned set of objects rather than Map.Entry objects
//  forgot to change result of Class.getName() from '.' delimiters to '/' delimiters
//  in compiler, direct-bind invokes didn't check if the target method was compiled yet
//  interleaving trimming and compiling was wrong.  we need to trim all classes, then compile.
//  in trim, forgot to reset field and method offsets to INVALID_OFFSET
//  in trim, forgot to reset overrides flag for virtual methods
//  accidentally reversed isConstant test in jq_StaticField
//  off-by-one error in check to grow array in x86CodeBuffer
//  cut-and-paste error: forgot to update assertion about instruction length in x86
//  forgot to initialize address field in ObjectMap.Entry
//  in trim, forgot to add vtable to root set
//  cut-and-paste error: shift amounts for writing array elements
//  in bootstrapper, need to traverse classes even if they are never instantiated
//  in trimmer, forgot to load classes for getfield/putfield ops
//  in trimmer, forgot to add root method to necessary member set
//  in bootstrapper, java/lang/Object static fields were allocated first
//  in bootstrapper, forgot to initialize static fields
//  in trimmer, forgot to add current static field values as instantiated types
//  in trimmer, not only current static field values, but everything reachable from them are instantiated
//  debugmsg takes byte array, not char array.
//  cut-and-paste error; writing to object arrays shifted by 3 instead of 2
//  forgot to initialize vtable after compiling instance methods
//  compared to 0 instead of INVALID_OFFSET in InstanceMethod.isVirtual
//  native routines should be marked __stdcall
//  in assembler, forgot to change endianness of 16 bit constants
//  vtable entries for inherited methods were not getting initialized
//  when backpatching forward branches, the pointer points after the site to be branched, rather than before
//  +/- error when calculating backward branch offsets
//  when calling Unsafe methods, codegen was assuming it needed to push the return values
//  reversed the order of the arguments in compilation of Unsafe.poke
//  SHIFT_ONCE value was incorrect
//  field offsets were all off by OBJ_HEADER_SIZE
//  conditional near backward jumps were off by 1 byte (instruction length 6, not 5)
//  put assertion in the wrong place in UTF8.isBootstrapJavaLang
//  forgot to update jq_ClassLoader.getType when adding classlib bootstrap support
//  UTF8.isBootstrapJavaLang test was not robust enough
//  forgot to add hijack code when using reflection in Trimmer and Bootstrapper
//  forgot to add fallback to hijack code when using reflection
//  cut-and-paste error: incorrectly undid hijacking when getting value during getstatic of jq_Method type
//  bug in UTF8.isValidMethodDescriptor, leading to infinite loop
//  did isAssignable test in Trimmer on field type without checking if the type was loaded
//  cut-and-paste error: in BytecodeVisitor, if_icmpne was doing CMP_EQ test
//  when building vtable, checking superclasses used getMember rather than getVirtualMethod, causing vtable to be incorrect
//  order of ops with << in invokeinterface
//  in checkcast/instance_of, reversed arguments of call to isAssignable
//  off-by-one error: forgot that invokeinterface runtime routine took an extra argument
//  trimmed interface types by checking against instantiated types, which is wrong
//  base object was wrong when calling getVirtualMethod when looking up interface methods in Trimmer
//  calculated address of arguments, but forgot to do a "peek" in invokeinterface
//  forgot to return value from invokeinterface
//  fixed typo in Unsafe.getThreadBlock read from FS[14]
//  cut-and-paste error: method exits were guarded by TraceBytecodes instead of TraceMethods
//  arguments pushed to call to arraystorecheck were incorrect
//  forgot to add System.in/out/err as necessary members
//  need to map java.lang.Class objects to jav_.lang.Class objects during bootstrapping, because they don't have field jq_type
//  likewise, need to map java.lang.ClassLoader objects to jav_.lang.ClassLoader
//  sun.io.CharToSingleByteConverter was being created with reflection and therefore Trimmer didn't catch it
//  forgot to change jq_Initializer and jq_ClassInitializer when adding VOID as a type
//  when jikes compiles static class references, it puts an "L" at the start of the name, so it wasn't finding the field with reflection.
//  skipped transient primitive fields caused a null pointer exception when writing their values
//  forgot to subtract from startAddress in some of the poke() methods in BootImage
//  calls to Unsafe were not successfully being bypassed in Trimmer (break; -> return;)
//  forgot to add fields/methods from Unsafe as necessary, leading them to being trimmed and a crash in bytecode traversal
//  code to trim references in constant pool replaced references with "jq_NameAndDesc" rather than "jq_MemberReference"
//  accidentally added Unsafe.<clinit> to the necessary method list in Trimmer
//  compiled_methods should not be a null static field
//  0xc001d00d was not causing a hardware trap when dereferenced!
//  stupid typo in debugmsg(String), was calling itself rather than debugmsg(byte[])
//  forgot to initialize jq_Type.class_object field during bootstrap
//  in ObjectTraverser, forgot to add support for arrays/primitive types when mapping between jq_Type and Class objects
//  primitive types use Integer.TYPE, not Integer.class
//  __stdcall has arguments in reverse order on stack! changed SystemInterface to match.
//  we can't actually skip transient fields (ArrayList.elementData is transient)
//  can't use Integer to represent code locations, as Integer.compareTo throws a ClassCastException
//  reversed the cases in jq_CompiledCode.compareTo, so it was never finding anything.
//  problem when Object or x86ReferenceLinker contains a native or abstract method, because during compilation it recursively tries to initialize itself.
//  adding compiled code when using bogus remapper crashes because the addresses are the same.  disabled compilation during bogus remapping.
//  confused order of reflective interpreter stack -> grows up, sp points to empty space
//  accidently shifted getParamWords left by 2, causing the interpreter to crash
//  forgot to check for null exception type in exception handling (from finally regions)
//  forgot to add a 'return' when calling object initializer during interpretation, leading to stack underflow
//  forgot to set thisptr param when calling getfield in reflective interpreter
//  problem with trying to interpret abstract method. added assertions so that it will be easier to catch next time.
//  forgot to roll back to previous state in interpreter when a method doesn't catch an exception
//  variable assignment bug in reflective interpreter invokehelper, assigned to m rather than f
//  can't call Thread.interrupt0 reflectively, so we skip over it in the reflective interpreter.
//  the reflective interpreter was crashing on checkcast/instanceof of null
//  correction to above- interpreter wasn't calling ANY private methods correctly. changed to use "getDeclaredMethod" rather than "getMethod"
//  forgot to add fields to ObjectTraverser after adding them to JDK classes
//  i2l is broken, it reversed the hi and lo words
//  lcmp/fcmp/dcmp were broken, the branch was off by one so less than returned equals
//  long shifts were incorrect, invalid instruction for AND_r_i8
//  bug in transcribing uqdivrem function: misread "1" as "i"
//  bug in x86CodeBuffer: when generated method overflowed the buffer, it initialized the counter to 0 rather than -1.  Manifested as incorrect backpatching.
//  frem/drem were incorrect because loads pushed the values on the stack, so they got reversed.
//  left a value on the fp register stack when doing fcmp/dcmp/frem/drem, which eventually caused later FP operations to mysteriously return NaN
//  in Trimmer, forgot to load/verify/prepare array classes from subarrays created with multinewarray
//  cut-and-paste error: BytecodeVisitor for LASTORE actually called visitIASTORE
//  arrayStoreCheck did not check for null before doing a getTypeOf on the object
//  getJavaLangObject() didn't work, probably because of some incorrect field masking behavior. renamed it to prim_desc2type.
//  forgot to add a base case to recursion in multinewarray_helper
//  in compiler, forgot to pop args after call to multinewarray runtime function
//  in multinewarray, looking up of varargs didn't take into account the two arguments already there
//  in multinewarray, args were used in reverse order
//  calculating the branch target in tableswitch was off by 4
//  f2i/f2l/d2i/d2l were rounding the incorrect way for negative values
//  used the wrong offset when loading back fp control word
//  used wrong mask for rounding mode in fp control word (0x0c00 not 0x60)
//  f2i/f2l/d2i/d2l did not handle exceptional cases (too big, too small, NaN) so I added explicit checks
//  forgot to free floating point registers in f2i/f2l/d2i/d2l exceptional cases
//  checking for catch blocks checked absolute ip's rather than offsets from code start, causing exceptions not to work
//  long-standing bug in Trimmer, addSuperclassVirtualMethods didn't work at all because the loop condition was reversed (c==null vs. c!=null)
//  bug in exception deliverer, order of operations: x<<2+4 is x<<(2+4) not (x<<2)+4
//  can't use JMath as replacement for StrictMath because it messes up bootstrapping, so moved to a separate class and made them call it
//  after bootstrapping, Class objects referred to the old system classloader.  fixed objecttraverser to map old classloader to the new one.
//  primordial classloader parent field referred to itself, leading to infinite recursion.
//  findBootstrapClass added L...; to the class name twice.
//  trycatch exception types were never actually being set (parameter name was "extype" rather than "exType")
//  did a +4 rather than a -4 when setting the exception object when branching to a catch block
//  looking up methods by ip didn't work if the ip pointed to the end address.  changed >= to >.
//  file open/close were not passing the file names as null-terminated byte strings
//  bug in exception handler range check, the given offset refers to the instr after the one that threw the exception, so it should check (low,high], not [low,high)
//  in reflective interpreter stack, popping longs was incorrect (did ++ instead of --)
//  in interpreter, lcmp2 pushes int, not long
//  in interpreter, lshift had the order of the two arguments reversed
//  cut-and-paste error: in BytecodeVisitor, dstore was actually calling visitLSTORE
//  forgot to load/verify/prepare exception types before doing type checks on them
//  bug in synchronized hack in reflective interpreter: synchronized methods executed twice because monitorenter also executes them
//  in reflective interpreter, exiting from a synchronized method didn't revert the state back to the old one.
//  passing double parameters in reflective interpreter was broken, because it pushed the double value of the long bits, rather than the real double value.
//  typo: interpreter was comparing FCMP2/DCMP2 arguments to "CMP_LT" rather than "CMP_L", leading it to always do the "CMP_G" case
//  forgot to redirect Unsafe.getTypeOf in reflective interpreter
//  forgot to redirect getJavaLangClassObject in reflective interpreter
//  scratch the last one, getfield/getstatic now all go through the ObjectTraverser
//  in FileInputStream.readBytes (and FileOutputStream.writeBytes), bounds check was off-by-one (off+len-1, not off+len)
//  FileInputStream.read was returning the return code from the read() call, rather than the actual byte read
//  files were being opened in text mode (default), causing reading to stop at the first ^Z character
//  when a class is dynamically loaded, it may refer to methods/fields that were trimmed. oops!
//  typo: field in ClassLoader was named _jq_type, rather than _desc2type
//  typo: poke1 signature was (II)V, instead of (IB)V
//  passed wrong flag into Trimmer (!TrimAllTypes == AddAllClassMembers)
//  forgot to load/verify/prepare classes of methods added to the worklist in Trimmer
//  forgot to redirect fields of java.util.zip.ZipFile after redirecting the class to org.jos.
//  using reflection on java.lang.ref.Finalizer objects is bad.  they change behind our backs.
//  jq_StaticMethod/jq_StaticField.needsDynamicLink was incorrect.  it should have been calling jq_Class.needsDynamicLink.
//  some of the patch_ routines in x86ReferenceCompiler were patching with poke4 rather than poke2
//  forgot to add "L" and ";" to class name of main class to load.
//  had the wrong descriptor when searching for main method (forgot '[')
//  reversed the order of code offset, bc index when building jq_BytecodeMap in reference compiler
//  typo in linker: loaded class "_class" rather than class "k"
//  when heap overflows, forgot to add object size to heapCurrent after allocating new block.  caused seemingly random but completely repeatable crashes.
//  patching invokestatic calls patched the absolute target address, not the relative offset to the target address.
//  LineNumberTable is an attribute of Code, not of MethodInfo
//  forgot to set jq_LineNumberBC as Comparable
//  incorrectly assumed that instance methods never needed dynamic links (if the target method is not compiled, it is necessary)
//  patch_invokevirtual instructions were incorrect
//  backpatching assumed retloc was at the end of the backpatch region
//  forgot to update return values from patch routines to reflect that retloc was not at the end of the backpatch region
//  in reference compiler, can't assert that a method is an interface until the class is loaded.
//  forgot to load/verify/prepare subarray types in isAssignable
//  forgot to cls_initialize when doing newarray
//  accidently was passing "args" to main, instead of "main_args", so the args included the main class name
//  when running under win2000, calloc would return memory with a lower address than the boot image memory, messing up exception delivery
//  forgot that cls_init could recursively call itself if a superclass created an object of a subclass's type in its clinit
//  during startup, primitive descriptors may not yet be registered in desc2type, so we need to set primitive-typed field descriptors after registering the primitive types.
//  incorrectly assumed that AllocEnabled in BootImage implied that Class objects would be allocated when doing state changes.
//  accidentally used abs rather than EA when emitting code for cas4
//  bug in putfield4 backpatch: was doing 8 bit version of instruction
//  typo: when growing array in Win32FileSystem.list, did "s2 = s" rather than "s = s2"
//  forgot to replace '.' with '/' when building descriptor for main class
//  mixed up the from and to arguments in memcpy
//  cut-and-paste bug in assembler: emit2_DISP32_SEImm8 was setting DISP8, rather than DISP32.  manifested in iinc of large indexed local var.
//  wrong flags to _open when opening for writing
//  thought that read() returned 0 at EOF, but it actually returns 1 (because it reroutes through readBytes())
//  during bootstrapping, class initialization triggered Unsafe.getTypeOf() in an assertion before the remapper was installed
//  cycle in dependencies for clinit, leading to null pointer exceptions because fields weren't initialized yet
//  forgot a ; in class descriptor, leading to getStaticField to attempt to check superclasses, leading to load being called recursively.
//  forgot to skip class initializers when merging static methods from one class into another
//  forgot to update mirror ClassLoader static methods to have the jq_Class as the first argument
//  when merging classes, we can't use the state variable from the class anymore, each member needs its own copy.
//  forgot to initialize state variable to STATE_LOADED when creating bridge methods to merged class
//  forgot to update mirror Class static methods to have the jq_Class as the first argument
//  accidentally was making more than one copy of some primitive types, because creating the primitive types triggered jq_Primitive to be clinited, so when we returned they had already been created.
//  accidentally put "java/lang/Map" rather than "java/util/Map" for a field type descriptor
//  jq_Class objects for nonexistent mirror classes were sticking around and their class_object fields couldn't be initialized because the classes don't exist.
//  cut-and-paste bug: forgot to update class descriptor in java.util.zip.ZipFile/ZipEntry
//  forgot a return statement after finding the right field in putInstanceFieldValue
//  slight problem: initializing a ZipFile causes ZipEntry to be created reflectively, but there is no no-arg constructor.  added explicit check for bootstrapping in ZipFile.
//  forgot to update constructors in Thread mirror class
//  forgot to add _class field to java.lang.reflect.Array
//  was creating a mirror class for an anonymous class in ZipFile.  changed to use a different class name.
//  FileSystem.getFileSystem cannot use java.io.FileSystem in its descriptor because it is not a public class, but we still need it to override correctly.
//  references to a merged method in other class's constant pools were not referring to the new, merged copy.  changed to mutate the old copy.
//  search/replace bug:  Thread.registerNatives got renamed to registerReflections
//  forgot to reset access_flags (specifically, native flag) when merging methods in classes
//  forgot to call _umask, so all files were being opened read-only
//  miscopied constants from C header file (octal, not hex!) leading to files being opened read-only
//  in Trimmer, can't just set AddAllClassMembers flag to false when we want to not automatically add all class methods, because we still want to add the fields.
//  in Trimmer, adding a static field by simply being in the same class did not add its value
//  don't call addStaticFieldValue on a primitive type field!
//  x86ReferenceCompiler always attempts to get the bytecodes of a method on creation, leading to crashes if the method doesn't have any bytecode
//  forgot to move initCallPatches call after the cls_initialize calls in Bootstrapper
//  jq_Method.compile_stub wasn't guaranteeing that compile() was compiled
//  changing to use Reflection class messed up reflection target prediction hack in Trimmer
//  Linker.invoke* were not being marked as necessary.
//  needsDynamicLink of instance fields/methods was not taking into account boot image types, leading all invokespecials to be dynamically linked
//  compile() of a native/abstract method was calling cls_initialize on x86ReferenceLinker (and therefore java.lang.Object) leading to extra call patches and corruption of code
//  compiling NEWARRAY caused cls_initialize to be called on the array type, which caused java.lang.Object methods to be compiled
//  forgot to add Reflection.obj_trav to the list of null static fields
//  setting jq.Bootstrapping should occur before any class initialization takes place (i.e. before setting trace flags)
//  Reflection._invokestatic_noargs was referring to nonexisting method
//  ZipEntry._constructor was referring to nonexisting method
//  CodeAllocator._compiled_methods had the wrong type signature
//  forgot that allocateCodeBlock code_reloc and data_reloc could be null, so the addAll call was crashing
//  BootstrapCodeAllocator corrected by startIndex and then called get1, which also corrected by startIndex
//  malformed loop: put return in loop rather than after it.
//  mistyped an offset when outputting the coff file
//  jq_Member.isFinal was checking the ACC_STATIC flag rather than the ACC_FINAL flag
//  inverted the section numbers in relocations (0 <-> 1)
//  forgot to relocate references to addresses of static fields in code
//  in backpatching, did target-code+4 instead of target-code-4, leading backpatchs to be off by 8
//  forgot to add relocs for vtable pointers in objects
//  wasn't adding relocs for external references
//  symbol table numbers were off by 2 because of inserted entry_0 and trap_handler_8 syments
//  name match against entry_0 and trap_handler_8 forgot numbers
//  forgot to skip adding relocs for null static fields
//  forgot to add jq_CompiledCode.entrypoint to code fields
//  was adding code relocations in the data segment to a list, but wasn't doing anything with that list
//  forgot some classes referenced by System.initializeSystemClass in the class list
//  pushing null constant caused relocation to be created for it
//  code (.text) section is by default write protected, so backpatching crashed
//  forgot to add some classes to the class list
//  forgot to initialize CodeAllocator.DEFAULT
//  forgot to relocate CodeAllocator.lowAddress and CodeAllocator.highAddress, so stack walking was broken.
//  CodeAllocator.lowAddress and CodeAllocator.highAddress were being initialized BEFORE anything was compiled!
//  by putting static field relocs in initStaticField, when initStaticField was called more than once for a field, relocs were being also added more than once
//  array classes not in list were not being initialized
//  comparison in CompiledCode.compareTo assertion was using > rather than >=
//  RuntimeCodeAllocator.endian2 was incorrect
//  Reflection.invoke were asserting for isClsInitialized, but static methods could be called during clinit
//  jq_InstanceMethod.needsDynamicLink was incorrect, condition was reversed
//  printing some floats crashed because f2i was not correct, because double constants were stored in the wrong dword order
//  in BootstrapCodeAllocator, was updating current_bundle ptr on ensureCapacity
//  forgot to support path seperator in command line class path
//  forgot to add padding to the last object in the boot image
//  in bootstrapper, adding reloc for static field failed because it was in nullStaticFields
//  in Reflection.putfield_L, wrong endianness
//  forgot to support path seperator in bootstrap command line class path
//  typo in ZipFile, used _class.newInstance() when I meant ZipEntry._class.newInstance()
//  forgot to trigger class loading before type checking when using Class.isAssignableFrom, etc.
//  tricky! in Utf8, table[getID()] failed because getID() could rewrite table, but the code was using the old table!!
//  calling keySet() on a TreeMap (compiled_methods) causes its fields to change, messing up bootstrapping.
//  system also creates an instance of ByteToChar via reflection.  added to trimmer.
//  exception delivery crashed when a non-Java stack frame was on the stack, because the cc had no method or TryCatch
//  forgot to add code/data relocs to generated cc for compile_stub
//  flipped assertion in ThreadQueue.enqueue
//  reversed the arguments in asm.emit2_Reg_Mem
//  disable/enableThreadSwitch in jq_Thread was a static function, causing it to always operate on the current thread!
//  forgot to push result in _isEQ, causing the stack to screw up (and isEQ to always return true)
//  was subtracting 1 instead of LOCK_COUNT_INC in monitorexit
//  in NativeThread, reallocated CodeAllocator rather than calling init() on it
//  status flags used 0xf, but Monitors were aligned at an 8 byte alignment
//  forgot to resume the timer interrupt thread after creating it!
//  run method of timer interrupt thread was not compiled and the thread block was not set yet, so it couldn't be compiled.
//  forgot to initialize code/heap allocator for interrupter native thread.
//  in threadSwitch, the register state pointed to the start of the threadSwitch function, so when we resumed from it, it just reentered the threadSwitch
//  compare and swap when installing inflated lock was incorrect.
//  in static synch methods, forgot to use Reflection.getJDKType rather than jq_Type.getJavaLangClassObject (which returns null during bootstrapping)
//  typo: _putstatic4 was actually pointing to _getstatic4
//  lock overflow was incorrect, was waiting for lock to be released.
//  when using LOCK_COUNT_MASK when lock entry count overflowed, forgot to right shift it.
//  forgot to set jq_RegisterState.ContextFlags before calling get_thread_context in ctrl_break_handler
//  getClassInitializer was using getStaticMethod instead of getDeclaredStaticMethod, leading to the superclass's <clinit> being invoked
//  using reflection to get a field object returned a new one on every call, causing the bootstrapping phase to screw up
//  forgot a k = k.getSuperclass() in a loop in getStaticFields, leading to an infinite loop
//  allocating arrays that were larger than the threshold didn't work.
//  in Sun's impl, readBytes/writeBytes doesn't throw an IndexOutOfBoundsException when offset == b.length and length == 0 (a bug on their part)
//  forgot to call load() in ClassLoader.findLoadedClass
//  off-by-one errors (too small) in stack walking in various situations. forgot to count the dummy frame from the native stub.
//  isInSamePackage predicate in String.equals was reversed
//  instance field lookup was using instance_fields array when it should have done a recursive lookup on declared_instance_fields
//  overlapping array case in arraycopy was backwards
//  forgot to check array bounds before copying in arraycopy.  the array would get written to even if the bounds were bad.
//  when doing setLocation for BytecodeVisitor, need to change i_end as well as i_start, because get___() routines use i_end
//  off-by-one error in BitStringIterator.hasNext
//  put a "this" instead of a "that" in copy constructor for TypeAnalysis.AnalysisState
//  forgot to set done=true in SingletonIterator, leading to infinite iteration
//  in TypeAnalysis.AnalysisSummary copy routine, mixed up this and that
//  in CFG builder, forgot to add ATHROW as branch connecting to exit
//  forgot to initialize stack depth in TypeAnalysis visitor
//  forgot to handle RET in basic block builder
//  in TypeAnalysis union, mixed up "this" and "that"
//  in interprocedural matching in TypeAnalysis, was calling "copy_deep" for outside nodes, leading to callee outside edges being matched against inside edges added from callee
//  in TypeAnalysis when copying using old_to_new map, we were recursive calling before adding to the map, leading to infinite recursion on cycles
//  with cycle detection in TypeAnalysis union, OutsideProgramLocation.union_deep pushed on the stack and then called supertype ProgramLocation.union_deep, which thought there was a cycle
//  forgot that astore can store jsr ret addr, leading to nullptr in TypeAnalysis
//  visitBytecode call relies on i_end being i_start-1.  array bounds exception
//  reversed the order of arrays in arraycopy, so the old array was getting cleared
//  typo when changing StaticField.sf_initialize, added "address" instead of "offset"

// TODO:
//  check if 0x80000000 / -1 works
//  check if x / 0L works

public abstract class jq {

    public static void boot() throws Throwable {
        try {
            // initialize the thread data structures, allocators, etc.
            jq_NativeThread.initInitialNativeThread();

            // init the ctrl-break handler thread.
            jq_NativeThread.initBreakThread();

            // init the garbage collector thread & set it as daemon
            jq_NativeThread.initGCThread();

            // call java.lang.System.initializeSystemClass()
            ClassLibInterface.DEFAULT.initializeSystemClass();
        } catch (Throwable x) {
            SystemInterface.debugmsg("Exception occurred during virtual machine initialization");
            SystemInterface.debugmsg("Exception: " + x);
            if (System.err != null) x.printStackTrace(System.err);
            return;
        }
        int numOfArgs = SystemInterface.main_argc();
        String[] args = new String[numOfArgs];
        for (int i = 0; i < numOfArgs; ++i) {
            int len = SystemInterface.main_argv_length(i);
            byte[] b = new byte[len];
            SystemInterface.main_argv(i, b);
            args[i] = new String(b);
        }
        String classpath = ".";
        int i = 0;
        for (; ;) {
            if (i == args.length) {
                printUsage();
                return;
            }
            if (args[i].equals("-cp") || args[i].equals("-classpath")) { // class path
                classpath = args[++i];
                ++i;
                continue;
            }
            if (args[i].equals("-nt") || args[i].equals("-native_threads")) { // class path
                NumOfNativeThreads = Integer.parseInt(args[++i]);
                ++i;
                continue;
            }
            // todo: other command line switches to change VM behavior.
            int j = TraceFlags.setTraceFlag(args, i);
            if (i != j) {
                i = j;
                continue;
            }
            break;
        }
        if (on_vm_startup != null) {
            Iterator it = on_vm_startup.iterator();
            while (it.hasNext()) {
                MethodInvocation mi = (MethodInvocation) it.next();
                try {
                    mi.invoke();
                } catch (Throwable x) {
                    SystemInterface.debugmsg("Exception occurred during virtual machine initialization");
                    SystemInterface.debugmsg(mi.toString());
                    SystemInterface.debugmsg("Exception: " + x);
                    x.printStackTrace(System.err);
                    return;
                }
            }
        }
        if (classpath != null) {
            Iterator it = PrimordialClassLoader.classpaths(classpath);
            while (it.hasNext()) {
                String s = (String) it.next();
                PrimordialClassLoader.loader.addToClasspath(s);
            }
        }

        jq_Thread tb = Unsafe.getThreadBlock();
        jq_NativeThread nt = tb.getNativeThread();
        jq_NativeThread.initNativeThreads(nt, NumOfNativeThreads);

        // Here we start method replacement of classes whose name were given as arguments to -replace on the cmd line.
        if (Clazz.jq_Class.TRACE_REPLACE_CLASS) SystemInterface.debugmsg("\nSTARTING REPLACEMENT of classes: " + Clazz.jq_Class.classToReplace);

        for (Iterator it = Clazz.jq_Class.classToReplace.iterator(); it.hasNext();) {
            String newCName = (String) it.next();
            PrimordialClassLoader.loader.replaceClass(newCName);
        }
        if (Clazz.jq_Class.TRACE_REPLACE_CLASS) SystemInterface.debugmsg("\nDONE with Classes Replacement!");

        String className = args[i];
        jq_Class main_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("L" + className.replace('.', '/') + ";");
        main_class.load();
        jq_StaticMethod main_method = main_class.getStaticMethod(new jq_NameAndDesc(Utf8.get("main"), Utf8.get("([Ljava/lang/String;)V")));
        if (main_method == null) {
            System.err.println("Class " + className + " does not contain a main method!");
            return;
        }
        if (!main_method.isPublic()) {
            System.err.println("Method " + main_method + " is not public!");
            return;
        }
        main_class.verify();
        main_class.prepare();
        main_class.sf_initialize();
        main_class.cls_initialize();
        String[] main_args = new String[args.length - i - 1];
        System.arraycopy(args, i + 1, main_args, 0, main_args.length);

        //jq_CompiledCode main_cc = main_method.getDefaultCompiledVersion();
        //Reflection.invokestatic_V(main_method, main_args);
        jq_MainThread mt = new jq_MainThread(main_method, main_args);
        mt.start();
        jq_NativeThread.startNativeThreads();
        nt.nativeThreadEntry();
        jq.UNREACHABLE();
    }

    public static void printUsage() {
        System.out.println("Usage: joeq <classname> <parameters>");
    }

    public static void initializeForHostJVMExecution() {
        jq.Bootstrapping = true;
        jq.DontCompile = true;
        jq.boot_types = new java.util.HashSet();

        CodeAddress.FACTORY = new CodeAddress.CodeAddressFactory() {
            public int size() {
                return 4;
            }
        };
        HeapAddress.FACTORY = new HeapAddress.HeapAddressFactory() {
            public int size() {
                return 4;
            }

            //public HeapAddress getNull() { return new Bootstrap.BootstrapHeapAddress(0); }
            //public HeapAddress addressOf(Object o) { return new Bootstrap.BootstrapHeapAddress(0); }
            //public HeapAddress address32(int val) { return new Bootstrap.BootstrapHeapAddress(val); }
            public HeapAddress getNull() {
                return null;
            }

            public HeapAddress addressOf(Object o) {
                return null;
            }

            public HeapAddress address32(int val) {
                return null;
            }
        };
        StackAddress.FACTORY = new StackAddress.StackAddressFactory() {
            public int size() {
                return 4;
            }

            public StackAddress alloca(int a) {
                jq.UNREACHABLE();
                return null;
            }

            public StackAddress getBasePointer() {
                jq.UNREACHABLE();
                return null;
            }

            public StackAddress getStackPointer() {
                jq.UNREACHABLE();
                return null;
            }
        };
        String classpath = System.getProperty("sun.boot.class.path") + System.getProperty("path.separator") + System.getProperty("java.class.path");
        for (Iterator it = PrimordialClassLoader.classpaths(classpath); it.hasNext();) {
            String s = (String) it.next();
            PrimordialClassLoader.loader.addToClasspath(s);
        }

        Reflection.obj_trav = ClassLib.Common.Interface.CommonObjectTraverser.INSTANCE;
        Reflection.obj_trav.initialize();
    }

    public static int NumOfNativeThreads = 1;
    public static boolean Bootstrapping;
    public static boolean DontCompile;
    public static Set boot_types;
    public static List on_vm_startup;

    public static boolean isBootType(jq_Type t) {
        return boot_types.contains(t);
    }

    public static /*final*/ boolean SMP = true;

    public static void Assert(boolean b, String reason) {
        if (!b) {
            SystemInterface.debugmsg("Assertion Failure!");
            SystemInterface.debugmsg(reason);
            if (!jq.Bootstrapping) {
                new InternalError().printStackTrace();
            }
            SystemInterface.die(-1);
        }
    }

    public static void Assert(boolean b) {
        Assert(b, "");
    }

    public static void TODO(String s) {
        SystemInterface.debugmsg("TODO: " + s);
        if (!jq.Bootstrapping) {
            new InternalError().printStackTrace();
        }
        SystemInterface.die(-1);
    }

    public static void TODO() {
        SystemInterface.debugmsg("TODO");
        if (!jq.Bootstrapping) {
            new InternalError().printStackTrace();
        }
        SystemInterface.die(-1);
    }

    public static void UNREACHABLE(String s) {
        SystemInterface.debugmsg("UNREACHABLE: " + s);
        if (!jq.Bootstrapping) {
            new InternalError().printStackTrace();
        }
        SystemInterface.die(-1);
    }

    public static void UNREACHABLE() {
        SystemInterface.debugmsg("BUG! unreachable code reached!");
        if (!jq.Bootstrapping) {
            new InternalError().printStackTrace();
        }
        SystemInterface.die(-1);
    }

    //// converting bytes to other data types
    public static char twoBytesToChar(byte b1, byte b2) {
        return (char) ((b1 << 8) | (b2 & 0xFF));
    }

    public static short twoBytesToShort(byte b1, byte b2) {
        return (short) ((b1 << 8) | (b2 & 0xFF));
    }

    public static int twoCharsToInt(char c1, char c2) {
        return (c1 << 16) | c2;
    }

    public static int fourBytesToInt(byte b1, byte b2, byte b3, byte b4) {
        return (b1 << 24) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 8) | (b4 & 0xFF);
    }

    public static long eightBytesToLong(byte b1, byte b2, byte b3, byte b4, byte b5, byte b6, byte b7, byte b8) {
        int hi = fourBytesToInt(b1, b2, b3, b4);
        int lo = fourBytesToInt(b5, b6, b7, b8);
        return twoIntsToLong(lo, hi);
    }

    public static char twoBytesToChar(byte[] b, int i) {
        return twoBytesToChar(b[i], b[i + 1]);
    }

    public static short twoBytesToShort(byte[] b, int i) {
        return twoBytesToShort(b[i], b[i + 1]);
    }

    public static int fourBytesToInt(byte[] b, int i) {
        return fourBytesToInt(b[i], b[i + 1], b[i + 2], b[i + 3]);
    }

    public static long eightBytesToLong(byte[] b, int i) {
        return eightBytesToLong(b[i], b[i + 1], b[i + 2], b[i + 3], b[i + 4], b[i + 5], b[i + 6], b[i + 7]);
    }

    public static long twoIntsToLong(int lo, int hi) {
        return (((long) lo) & 0xFFFFFFFFL) | ((long) hi << 32);
    }

    public static void charToTwoBytes(char i, byte[] b, int index) {
        b[index] = (byte) (i >> 8);
        b[index + 1] = (byte) (i);
    }

    public static void intToFourBytes(int i, byte[] b, int index) {
        b[index] = (byte) (i >> 24);
        b[index + 1] = (byte) (i >> 16);
        b[index + 2] = (byte) (i >> 8);
        b[index + 3] = (byte) (i);
    }

    public static void longToEightBytes(long i, byte[] b, int index) {
        b[index] = (byte) (i >> 56);
        b[index + 1] = (byte) (i >> 48);
        b[index + 2] = (byte) (i >> 40);
        b[index + 3] = (byte) (i >> 32);
        b[index + 4] = (byte) (i >> 24);
        b[index + 5] = (byte) (i >> 16);
        b[index + 6] = (byte) (i >> 8);
        b[index + 7] = (byte) (i);
    }

    //// useful string utility functions
    public static String hex(int i) {
        return "0x" + Integer.toHexString(i);
    }

    public static String hex(Object o) {
        if (jq.Bootstrapping)
            return hex(System.identityHashCode(o));
        else
            return HeapAddress.addressOf(o).stringRep();
    }

    public static String hex8(int i) {
        String t = Integer.toHexString(i);
        return "0x00000000".substring(0, 10 - t.length()) + t;
    }

    public static String hex16(long i) {
        String t = Long.toHexString(i);
        return "0x0000000000000000".substring(0, 18 - t.length()) + t;
    }

    public static String shex(int i) {
        if (i < 0)
            return "-" + hex(-i);
        else
            return hex(i);
    }

    public static String left(String s, int w) {
        int n = s.length();
        if (w < n) return s.substring(0, w);
        StringBuffer b = new StringBuffer(w);
        b.append(s);
        for (int i = n; i < w; ++i) {
            b.append(' ');
        }
        return b.toString();
    }

    public static String right(String s, int w) {
        int n = s.length();
        if (w < n) return s.substring(n - w);
        StringBuffer b = new StringBuffer(w);
        for (int i = n; i < w; ++i) {
            b.append(' ');
        }
        b.append(s);
        return b.toString();
    }

    //// useful functions for parsing class and method names
    public static jq_Type parseType(String s) {
        if (s.length() == 1) {
            jq_Primitive t = (jq_Primitive) PrimordialClassLoader.loader.getBSType(s);
            if (t != null) return t;
            s = "L" + s + ";";
        } else {
            s = s.replace('.', '/');
            int arrayDepth = 0;
            while (s.endsWith("[]")) {
                ++arrayDepth;
                s = s.substring(0, s.length() - 2);
            }
            if (!s.startsWith("[") && !s.endsWith(";"))
                s = "L" + s + ";";
            while (--arrayDepth >= 0)
                s = "[" + s;
        }
        return (jq_Reference) PrimordialClassLoader.loader.getOrCreateBSType(s);
    }

    //public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateType("LMain/jq;");
    //public static final jq_StaticMethod _hex8 = _class.getOrCreateStaticMethod("hex8", "(I)Ljava/lang/String;");
    //public static final jq_StaticMethod _hex16 = _class.getOrCreateStaticMethod("hex16", "(J)Ljava/lang/String;");
}
