/*
 * jq_ClassFileConstants.java
 *
 * Created on December 19, 2000, 9:02 AM
 *
 */

package Clazz;

/*
 * @author  John Whaley
 * @version $Id$
 */
public interface jq_ClassFileConstants {

    /* ClassFile format: common access flags for classes(C), methods(M) and fields(F)
       (VM Spec Chapter 4.7) */
    public static final char ACC_PUBLIC       = 0x0001; // C, M, F
    public static final char ACC_PRIVATE      = 0x0002; // M, F
    public static final char ACC_PROTECTED    = 0x0004; // M, F
    public static final char ACC_STATIC       = 0x0008; // M, F
    public static final char ACC_FINAL        = 0x0010; // C, M, F
    public static final char ACC_SYNCHRONIZED = 0x0020; // same value M, F
    public static final char ACC_SUPER        = 0x0020; // same value C
    public static final char ACC_VOLATILE     = 0x0040; // M  Declared volatile; cannot be cached
    public static final char ACC_TRANSIENT    = 0x0080; // M
    public static final char ACC_NATIVE       = 0x0100; // F
    public static final char ACC_INTERFACE    = 0x0200; // C
    public static final char ACC_ABSTRACT     = 0x0400; // C, F
    public static final char ACC_STRICT       = 0x0800; // F  Declared strictfp; floating-point mode is FP-strict

    /* Each item in the constant_pool table must begin with
       a 1-byte tag indicating the kind of cp_info entry.
       The contents of the info array vary with the value of tag.
       Each tag byte must be followed by two or more bytes
       giving information about the specific constant.
       The format of the additional information varies with
       the tag value. Tag values are given below.
       (VM Spec Chapter 4.4) */

    public static final byte CONSTANT_Class              = 7;
    public static final byte CONSTANT_FieldRef           = 9;
    public static final byte CONSTANT_MethodRef          = 10;
    public static final byte CONSTANT_InterfaceMethodRef = 11;
    public static final byte CONSTANT_String             = 8;
    public static final byte CONSTANT_Integer            = 3;
    public static final byte CONSTANT_Float              = 4;
    public static final byte CONSTANT_Long               = 5;
    public static final byte CONSTANT_Double             = 6;
    public static final byte CONSTANT_NameAndType        = 12;
    public static final byte CONSTANT_Utf8               = 1;
    public static final byte CONSTANT_ResolvedClass      = 13; // doesn't exist in class file.
    public static final byte CONSTANT_ResolvedSFieldRef  = 14; // doesn't exist in class file.
    public static final byte CONSTANT_ResolvedIFieldRef  = 15; // doesn't exist in class file.
    public static final byte CONSTANT_ResolvedSMethodRef = 16; // doesn't exist in class file.
    public static final byte CONSTANT_ResolvedIMethodRef = 17; // doesn't exist in class file.

    /* A descriptor is a string representing the type of a field
       or method. Descriptors are represented in the class file
       format using UTF-8 strings. Followings are the BaseType
       characters corresponding to respective types.
       (VM Spec Chapter 4.3) */
    public static final byte TC_BYTE     = (byte)'B'; // signed byte
    public static final byte TC_CHAR     = (byte)'C'; // Unicode character
    public static final byte TC_DOUBLE   = (byte)'D'; // double-precision floating-point value
    public static final byte TC_FLOAT    = (byte)'F'; // single-precision floating-point value
    public static final byte TC_INT      = (byte)'I'; // integer
    public static final byte TC_LONG     = (byte)'J'; // long integer
    public static final byte TC_CLASS    = (byte)'L'; // L<classname>; an instance of class <classname>
    public static final byte TC_CLASSEND = (byte)';'; //
    public static final byte TC_SHORT    = (byte)'S'; // signed short
    public static final byte TC_BOOLEAN  = (byte)'Z'; // true or false
    public static final byte TC_ARRAY    = (byte)'['; // one array dimension
    public static final byte TC_PARAM    = (byte)'('; //
    public static final byte TC_PARAMEND = (byte)')'; //
    public static final byte TC_VOID     = (byte)'V'; // indicates that the method returns no value(void)

    public static final byte T_BOOLEAN = 4;
    public static final byte T_CHAR    = 5;
    public static final byte T_FLOAT   = 6;
    public static final byte T_DOUBLE  = 7;
    public static final byte T_BYTE    = 8;
    public static final byte T_SHORT   = 9;
    public static final byte T_INT     = 10;
    public static final byte T_LONG    = 11;

    // We have seen a reference to this class/member (for example, in the constant
    // pool of another class), but it has not been loaded, and therefore we know
    // nothing about it other than its name.
    public static final byte STATE_UNLOADED     = 0;
    // A thread is in the process of loading the constant pool for this class.
    // (see verify pass 1 Jvm spec 4.9.1)
    public static final byte STATE_LOADING1     = 1;
    // A thread has finished loading the constant pool, and is loading the class
    // members and other information.
    public static final byte STATE_LOADING2     = 2;
    // A thread has finished loading the class, and is now merging in implementation
    // classes.
    public static final byte STATE_LOADING3     = 3;
    // This class has been loaded and all members have been created.
    public static final byte STATE_LOADED       = 4;
    // A thread is in the process of verifying this class. (Jvm spec 2.17.3)
    // (see verify pass 2 Jvm spec 4.9.1)
    // It checks the code in each declared method in the class.
    public static final byte STATE_VERIFYING    = 5;
    // This class has been successfully verified.
    public static final byte STATE_VERIFIED     = 6;
    // A thread is in the process of preparing this class. (Jvm spec 2.17.3)
    // Preparation lays out the object fields and creates a method table.
    // Static fields are created and initialized in the NEXT step.
    public static final byte STATE_PREPARING    = 7;
    // This class has been prepared.
    public static final byte STATE_PREPARED     = 8;
    // A thread is creating the static fields for the class, and initializing the
    // ones that have ConstantValue attributes.
    public static final byte STATE_SFINITIALIZING = 9;
    // This class has its static fields created and initialized.
    public static final byte STATE_SFINITIALIZED = 10;
    // A thread is in the process of initializing this class. (Jvm spec 2.17.4-5)
    // Initialization is triggered when code is about to execute that will create
    // an instance, execute a static method, or use or assign a nonconstant static
    // field.
    public static final byte STATE_CLSINITIALIZING = 11;
    public static final byte STATE_CLSINITRUNNING = 12;
    // An error occurred during initialization!  This resulted in a throwing of 
    // a NoClassDefFoundError, ExceptionInInitializerError, or OutOfMemoryError
    // for the initializing thread.  Any further attempts to initialize should
    // result in the throwing of a NoClassDefFoundError. 
    public static final byte STATE_CLSINITERROR = 13;
    // This class has been fully initialized!
    public static final byte STATE_CLSINITIALIZED  = 14;

    /**
     * Illegal codes
     */
    public static final short  UNDEFINED      = -1;
    public static final short  UNPREDICTABLE  = -2;
    public static final short  RESERVED       = -3;
    public static final String ILLEGAL_OPCODE = "<illegal opcode>";
    public static final String ILLEGAL_TYPE   = "<illegal type>";

    /** Java VM opcodes.
     *  (VM Spec Chapter 6)
     */
    public static final short jbc_NOP              = 0;
    public static final short jbc_ACONST_NULL      = 1;
    public static final short jbc_ICONST_M1        = 2;
    public static final short jbc_ICONST_0         = 3;
    public static final short jbc_ICONST_1         = 4;
    public static final short jbc_ICONST_2         = 5;
    public static final short jbc_ICONST_3         = 6;
    public static final short jbc_ICONST_4         = 7;
    public static final short jbc_ICONST_5         = 8;
    public static final short jbc_LCONST_0         = 9;
    public static final short jbc_LCONST_1         = 10;
    public static final short jbc_FCONST_0         = 11;
    public static final short jbc_FCONST_1         = 12;
    public static final short jbc_FCONST_2         = 13;
    public static final short jbc_DCONST_0         = 14;
    public static final short jbc_DCONST_1         = 15;
    public static final short jbc_BIPUSH           = 16;
    public static final short jbc_SIPUSH           = 17;
    public static final short jbc_LDC              = 18;
    public static final short jbc_LDC_W            = 19;
    public static final short jbc_LDC2_W           = 20;
    public static final short jbc_ILOAD            = 21;
    public static final short jbc_LLOAD            = 22;
    public static final short jbc_FLOAD            = 23;
    public static final short jbc_DLOAD            = 24;
    public static final short jbc_ALOAD            = 25;
    public static final short jbc_ILOAD_0          = 26;
    public static final short jbc_ILOAD_1          = 27;
    public static final short jbc_ILOAD_2          = 28;
    public static final short jbc_ILOAD_3          = 29;
    public static final short jbc_LLOAD_0          = 30;
    public static final short jbc_LLOAD_1          = 31;
    public static final short jbc_LLOAD_2          = 32;
    public static final short jbc_LLOAD_3          = 33;
    public static final short jbc_FLOAD_0          = 34;
    public static final short jbc_FLOAD_1          = 35;
    public static final short jbc_FLOAD_2          = 36;
    public static final short jbc_FLOAD_3          = 37;
    public static final short jbc_DLOAD_0          = 38;
    public static final short jbc_DLOAD_1          = 39;
    public static final short jbc_DLOAD_2          = 40;
    public static final short jbc_DLOAD_3          = 41;
    public static final short jbc_ALOAD_0          = 42;
    public static final short jbc_ALOAD_1          = 43;
    public static final short jbc_ALOAD_2          = 44;
    public static final short jbc_ALOAD_3          = 45;
    public static final short jbc_IALOAD           = 46;
    public static final short jbc_LALOAD           = 47;
    public static final short jbc_FALOAD           = 48;
    public static final short jbc_DALOAD           = 49;
    public static final short jbc_AALOAD           = 50;
    public static final short jbc_BALOAD           = 51;
    public static final short jbc_CALOAD           = 52;
    public static final short jbc_SALOAD           = 53;
    public static final short jbc_ISTORE           = 54;
    public static final short jbc_LSTORE           = 55;
    public static final short jbc_FSTORE           = 56;
    public static final short jbc_DSTORE           = 57;
    public static final short jbc_ASTORE           = 58;
    public static final short jbc_ISTORE_0         = 59;
    public static final short jbc_ISTORE_1         = 60;
    public static final short jbc_ISTORE_2         = 61;
    public static final short jbc_ISTORE_3         = 62;
    public static final short jbc_LSTORE_0         = 63;
    public static final short jbc_LSTORE_1         = 64;
    public static final short jbc_LSTORE_2         = 65;
    public static final short jbc_LSTORE_3         = 66;
    public static final short jbc_FSTORE_0         = 67;
    public static final short jbc_FSTORE_1         = 68;
    public static final short jbc_FSTORE_2         = 69;
    public static final short jbc_FSTORE_3         = 70;
    public static final short jbc_DSTORE_0         = 71;
    public static final short jbc_DSTORE_1         = 72;
    public static final short jbc_DSTORE_2         = 73;
    public static final short jbc_DSTORE_3         = 74;
    public static final short jbc_ASTORE_0         = 75;
    public static final short jbc_ASTORE_1         = 76;
    public static final short jbc_ASTORE_2         = 77;
    public static final short jbc_ASTORE_3         = 78;
    public static final short jbc_IASTORE          = 79;
    public static final short jbc_LASTORE          = 80;
    public static final short jbc_FASTORE          = 81;
    public static final short jbc_DASTORE          = 82;
    public static final short jbc_AASTORE          = 83;
    public static final short jbc_BASTORE          = 84;
    public static final short jbc_CASTORE          = 85;
    public static final short jbc_SASTORE          = 86;
    public static final short jbc_POP              = 87;
    public static final short jbc_POP2             = 88;
    public static final short jbc_DUP              = 89;
    public static final short jbc_DUP_X1           = 90;
    public static final short jbc_DUP_X2           = 91;
    public static final short jbc_DUP2             = 92;
    public static final short jbc_DUP2_X1          = 93;
    public static final short jbc_DUP2_X2          = 94;
    public static final short jbc_SWAP             = 95;
    public static final short jbc_IADD             = 96;
    public static final short jbc_LADD             = 97;
    public static final short jbc_FADD             = 98;
    public static final short jbc_DADD             = 99;
    public static final short jbc_ISUB             = 100;
    public static final short jbc_LSUB             = 101;
    public static final short jbc_FSUB             = 102;
    public static final short jbc_DSUB             = 103;
    public static final short jbc_IMUL             = 104;
    public static final short jbc_LMUL             = 105;
    public static final short jbc_FMUL             = 106;
    public static final short jbc_DMUL             = 107;
    public static final short jbc_IDIV             = 108;
    public static final short jbc_LDIV             = 109;
    public static final short jbc_FDIV             = 110;
    public static final short jbc_DDIV             = 111;
    public static final short jbc_IREM             = 112;
    public static final short jbc_LREM             = 113;
    public static final short jbc_FREM             = 114;
    public static final short jbc_DREM             = 115;
    public static final short jbc_INEG             = 116;
    public static final short jbc_LNEG             = 117;
    public static final short jbc_FNEG             = 118;
    public static final short jbc_DNEG             = 119;
    public static final short jbc_ISHL             = 120;
    public static final short jbc_LSHL             = 121;
    public static final short jbc_ISHR             = 122;
    public static final short jbc_LSHR             = 123;
    public static final short jbc_IUSHR            = 124;
    public static final short jbc_LUSHR            = 125;
    public static final short jbc_IAND             = 126;
    public static final short jbc_LAND             = 127;
    public static final short jbc_IOR              = 128;
    public static final short jbc_LOR              = 129;
    public static final short jbc_IXOR             = 130;
    public static final short jbc_LXOR             = 131;
    public static final short jbc_IINC             = 132;
    public static final short jbc_I2L              = 133;
    public static final short jbc_I2F              = 134;
    public static final short jbc_I2D              = 135;
    public static final short jbc_L2I              = 136;
    public static final short jbc_L2F              = 137;
    public static final short jbc_L2D              = 138;
    public static final short jbc_F2I              = 139;
    public static final short jbc_F2L              = 140;
    public static final short jbc_F2D              = 141;
    public static final short jbc_D2I              = 142;
    public static final short jbc_D2L              = 143;
    public static final short jbc_D2F              = 144;
    public static final short jbc_I2B              = 145;
    public static final short jbc_INT2BYTE         = 145; // Old notion
    public static final short jbc_I2C              = 146;
    public static final short jbc_INT2CHAR         = 146; // Old notion
    public static final short jbc_I2S              = 147;
    public static final short jbc_INT2SHORT        = 147; // Old notion
    public static final short jbc_LCMP             = 148;
    public static final short jbc_FCMPL            = 149;
    public static final short jbc_FCMPG            = 150;
    public static final short jbc_DCMPL            = 151;
    public static final short jbc_DCMPG            = 152;
    public static final short jbc_IFEQ             = 153;
    public static final short jbc_IFNE             = 154;
    public static final short jbc_IFLT             = 155;
    public static final short jbc_IFGE             = 156;
    public static final short jbc_IFGT             = 157;
    public static final short jbc_IFLE             = 158;
    public static final short jbc_IF_ICMPEQ        = 159;
    public static final short jbc_IF_ICMPNE        = 160;
    public static final short jbc_IF_ICMPLT        = 161;
    public static final short jbc_IF_ICMPGE        = 162;
    public static final short jbc_IF_ICMPGT        = 163;
    public static final short jbc_IF_ICMPLE        = 164;
    public static final short jbc_IF_ACMPEQ        = 165;
    public static final short jbc_IF_ACMPNE        = 166;
    public static final short jbc_GOTO             = 167;
    public static final short jbc_JSR              = 168;
    public static final short jbc_RET              = 169;
    public static final short jbc_TABLESWITCH      = 170;
    public static final short jbc_LOOKUPSWITCH     = 171;
    public static final short jbc_IRETURN          = 172;
    public static final short jbc_LRETURN          = 173;
    public static final short jbc_FRETURN          = 174;
    public static final short jbc_DRETURN          = 175;
    public static final short jbc_ARETURN          = 176;
    public static final short jbc_RETURN           = 177;
    public static final short jbc_GETSTATIC        = 178;
    public static final short jbc_PUTSTATIC        = 179;
    public static final short jbc_GETFIELD         = 180;
    public static final short jbc_PUTFIELD         = 181;
    public static final short jbc_INVOKEVIRTUAL    = 182;
    public static final short jbc_INVOKESPECIAL    = 183;
    public static final short jbc_INVOKENONVIRTUAL = 183; // Old name in JDK 1.0
    public static final short jbc_INVOKESTATIC     = 184;
    public static final short jbc_INVOKEINTERFACE  = 185;
    public static final short jbc_NEW              = 187;
    public static final short jbc_NEWARRAY         = 188;
    public static final short jbc_ANEWARRAY        = 189;
    public static final short jbc_ARRAYLENGTH      = 190;
    public static final short jbc_ATHROW           = 191;
    public static final short jbc_CHECKCAST        = 192;
    public static final short jbc_INSTANCEOF       = 193;
    public static final short jbc_MONITORENTER     = 194;
    public static final short jbc_MONITOREXIT      = 195;
    public static final short jbc_WIDE             = 196;
    public static final short jbc_MULTIANEWARRAY   = 197;
    public static final short jbc_IFNULL           = 198;
    public static final short jbc_IFNONNULL        = 199;
    public static final short jbc_GOTO_W           = 200;
    public static final short jbc_JSR_W            = 201;

    /**
     * Non-legal opcodes, may be used by JVM internally.
     */
    public static final short jbc_BREAKPOINT       = 202;

    /**
     * Number of byte code operands, i.e., number of bytes after the tag byte
     * itself.
     */
  public static final short[] NO_OF_OPERANDS = {
      0/*nop*/, 0/*aconst_null*/, 0/*iconst_m1*/, 0/*iconst_0*/,
      0/*iconst_1*/, 0/*iconst_2*/, 0/*iconst_3*/, 0/*iconst_4*/,
      0/*iconst_5*/, 0/*lconst_0*/, 0/*lconst_1*/, 0/*fconst_0*/,
      0/*fconst_1*/, 0/*fconst_2*/, 0/*dconst_0*/, 0/*dconst_1*/,
      1/*bipush*/, 2/*sipush*/, 1/*ldc*/, 2/*ldc_w*/, 2/*ldc2_w*/,
      1/*iload*/, 1/*lload*/, 1/*fload*/, 1/*dload*/, 1/*aload*/,
      0/*iload_0*/, 0/*iload_1*/, 0/*iload_2*/, 0/*iload_3*/,
      0/*lload_0*/, 0/*lload_1*/, 0/*lload_2*/, 0/*lload_3*/,
      0/*fload_0*/, 0/*fload_1*/, 0/*fload_2*/, 0/*fload_3*/,
      0/*dload_0*/, 0/*dload_1*/, 0/*dload_2*/, 0/*dload_3*/,
      0/*aload_0*/, 0/*aload_1*/, 0/*aload_2*/, 0/*aload_3*/,
      0/*iaload*/, 0/*laload*/, 0/*faload*/, 0/*daload*/,
      0/*aaload*/, 0/*baload*/, 0/*caload*/, 0/*saload*/,
      1/*istore*/, 1/*lstore*/, 1/*fstore*/, 1/*dstore*/,
      1/*astore*/, 0/*istore_0*/, 0/*istore_1*/, 0/*istore_2*/,
      0/*istore_3*/, 0/*lstore_0*/, 0/*lstore_1*/, 0/*lstore_2*/,
      0/*lstore_3*/, 0/*fstore_0*/, 0/*fstore_1*/, 0/*fstore_2*/,
      0/*fstore_3*/, 0/*dstore_0*/, 0/*dstore_1*/, 0/*dstore_2*/,
      0/*dstore_3*/, 0/*astore_0*/, 0/*astore_1*/, 0/*astore_2*/,
      0/*astore_3*/, 0/*iastore*/, 0/*lastore*/, 0/*fastore*/,
      0/*dastore*/, 0/*aastore*/, 0/*bastore*/, 0/*castore*/,
      0/*sastore*/, 0/*pop*/, 0/*pop2*/, 0/*dup*/, 0/*dup_x1*/,
      0/*dup_x2*/, 0/*dup2*/, 0/*dup2_x1*/, 0/*dup2_x2*/, 0/*swap*/,
      0/*iadd*/, 0/*ladd*/, 0/*fadd*/, 0/*dadd*/, 0/*isub*/,
      0/*lsub*/, 0/*fsub*/, 0/*dsub*/, 0/*imul*/, 0/*lmul*/,
      0/*fmul*/, 0/*dmul*/, 0/*idiv*/, 0/*ldiv*/, 0/*fdiv*/,
      0/*ddiv*/, 0/*irem*/, 0/*lrem*/, 0/*frem*/, 0/*drem*/,
      0/*ineg*/, 0/*lneg*/, 0/*fneg*/, 0/*dneg*/, 0/*ishl*/,
      0/*lshl*/, 0/*ishr*/, 0/*lshr*/, 0/*iushr*/, 0/*lushr*/,
      0/*iand*/, 0/*land*/, 0/*ior*/, 0/*lor*/, 0/*ixor*/, 0/*lxor*/,
      2/*iinc*/, 0/*i2l*/, 0/*i2f*/, 0/*i2d*/, 0/*l2i*/, 0/*l2f*/,
      0/*l2d*/, 0/*f2i*/, 0/*f2l*/, 0/*f2d*/, 0/*d2i*/, 0/*d2l*/,
      0/*d2f*/, 0/*i2b*/, 0/*i2c*/, 0/*i2s*/, 0/*lcmp*/, 0/*fcmpl*/,
      0/*fcmpg*/, 0/*dcmpl*/, 0/*dcmpg*/, 2/*ifeq*/, 2/*ifne*/,
      2/*iflt*/, 2/*ifge*/, 2/*ifgt*/, 2/*ifle*/, 2/*if_icmpeq*/,
      2/*if_icmpne*/, 2/*if_icmplt*/, 2/*if_icmpge*/, 2/*if_icmpgt*/,
      2/*if_icmple*/, 2/*if_acmpeq*/, 2/*if_acmpne*/, 2/*goto*/,
      2/*jsr*/, 1/*ret*/, UNPREDICTABLE/*tableswitch*/, UNPREDICTABLE/*lookupswitch*/,
      0/*ireturn*/, 0/*lreturn*/, 0/*freturn*/,
      0/*dreturn*/, 0/*areturn*/, 0/*return*/,
      2/*getstatic*/, 2/*putstatic*/, 2/*getfield*/,
      2/*putfield*/, 2/*invokevirtual*/, 2/*invokespecial*/, 2/*invokestatic*/,
      4/*invokeinterface*/, UNDEFINED, 2/*new*/,
      1/*newarray*/, 2/*anewarray*/,
      0/*arraylength*/, 0/*athrow*/, 2/*checkcast*/,
      2/*instanceof*/, 0/*monitorenter*/,
      0/*monitorexit*/, UNPREDICTABLE/*wide*/, 3/*multianewarray*/,
      2/*ifnull*/, 2/*ifnonnull*/, 4/*goto_w*/,
      4/*jsr_w*/, 0/*breakpoint*/, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
      UNDEFINED, UNDEFINED, RESERVED/*impdep1*/, RESERVED/*impdep2*/
  };

    /**
     * How the byte code operands are to be interpreted.
     */
    public static final short[][] TYPE_OF_OPERANDS = {
        {}/*nop*/, {}/*aconst_null*/, {}/*iconst_m1*/, {}/*iconst_0*/,
        {}/*iconst_1*/, {}/*iconst_2*/, {}/*iconst_3*/, {}/*iconst_4*/,
        {}/*iconst_5*/, {}/*lconst_0*/, {}/*lconst_1*/, {}/*fconst_0*/,
        {}/*fconst_1*/, {}/*fconst_2*/, {}/*dconst_0*/, {}/*dconst_1*/,
        {T_BYTE}/*bipush*/, {T_SHORT}/*sipush*/, {T_BYTE}/*ldc*/,
        {T_SHORT}/*ldc_w*/, {T_SHORT}/*ldc2_w*/,
        {T_BYTE}/*iload*/, {T_BYTE}/*lload*/, {T_BYTE}/*fload*/,
        {T_BYTE}/*dload*/, {T_BYTE}/*aload*/, {}/*iload_0*/,
        {}/*iload_1*/, {}/*iload_2*/, {}/*iload_3*/, {}/*lload_0*/,
        {}/*lload_1*/, {}/*lload_2*/, {}/*lload_3*/, {}/*fload_0*/,
        {}/*fload_1*/, {}/*fload_2*/, {}/*fload_3*/, {}/*dload_0*/,
        {}/*dload_1*/, {}/*dload_2*/, {}/*dload_3*/, {}/*aload_0*/,
        {}/*aload_1*/, {}/*aload_2*/, {}/*aload_3*/, {}/*iaload*/,
        {}/*laload*/, {}/*faload*/, {}/*daload*/, {}/*aaload*/,
        {}/*baload*/, {}/*caload*/, {}/*saload*/, {T_BYTE}/*istore*/,
        {T_BYTE}/*lstore*/, {T_BYTE}/*fstore*/, {T_BYTE}/*dstore*/,
        {T_BYTE}/*astore*/, {}/*istore_0*/, {}/*istore_1*/,
        {}/*istore_2*/, {}/*istore_3*/, {}/*lstore_0*/, {}/*lstore_1*/,
        {}/*lstore_2*/, {}/*lstore_3*/, {}/*fstore_0*/, {}/*fstore_1*/,
        {}/*fstore_2*/, {}/*fstore_3*/, {}/*dstore_0*/, {}/*dstore_1*/,
        {}/*dstore_2*/, {}/*dstore_3*/, {}/*astore_0*/, {}/*astore_1*/,
        {}/*astore_2*/, {}/*astore_3*/, {}/*iastore*/, {}/*lastore*/,
        {}/*fastore*/, {}/*dastore*/, {}/*aastore*/, {}/*bastore*/,
        {}/*castore*/, {}/*sastore*/, {}/*pop*/, {}/*pop2*/, {}/*dup*/,
        {}/*dup_x1*/, {}/*dup_x2*/, {}/*dup2*/, {}/*dup2_x1*/,
        {}/*dup2_x2*/, {}/*swap*/, {}/*iadd*/, {}/*ladd*/, {}/*fadd*/,
        {}/*dadd*/, {}/*isub*/, {}/*lsub*/, {}/*fsub*/, {}/*dsub*/,
        {}/*imul*/, {}/*lmul*/, {}/*fmul*/, {}/*dmul*/, {}/*idiv*/,
        {}/*ldiv*/, {}/*fdiv*/, {}/*ddiv*/, {}/*irem*/, {}/*lrem*/,
        {}/*frem*/, {}/*drem*/, {}/*ineg*/, {}/*lneg*/, {}/*fneg*/,
        {}/*dneg*/, {}/*ishl*/, {}/*lshl*/, {}/*ishr*/, {}/*lshr*/,
        {}/*iushr*/, {}/*lushr*/, {}/*iand*/, {}/*land*/, {}/*ior*/,
        {}/*lor*/, {}/*ixor*/, {}/*lxor*/, {T_BYTE, T_BYTE}/*iinc*/,
        {}/*i2l*/, {}/*i2f*/, {}/*i2d*/, {}/*l2i*/, {}/*l2f*/, {}/*l2d*/,
        {}/*f2i*/, {}/*f2l*/, {}/*f2d*/, {}/*d2i*/, {}/*d2l*/, {}/*d2f*/,
        {}/*i2b*/, {}/*i2c*/,{}/*i2s*/, {}/*lcmp*/, {}/*fcmpl*/,
        {}/*fcmpg*/, {}/*dcmpl*/, {}/*dcmpg*/, {T_SHORT}/*ifeq*/,
        {T_SHORT}/*ifne*/, {T_SHORT}/*iflt*/, {T_SHORT}/*ifge*/,
        {T_SHORT}/*ifgt*/, {T_SHORT}/*ifle*/, {T_SHORT}/*if_icmpeq*/,
        {T_SHORT}/*if_icmpne*/, {T_SHORT}/*if_icmplt*/,
        {T_SHORT}/*if_icmpge*/, {T_SHORT}/*if_icmpgt*/,
        {T_SHORT}/*if_icmple*/, {T_SHORT}/*if_acmpeq*/,
        {T_SHORT}/*if_acmpne*/, {T_SHORT}/*goto*/, {T_SHORT}/*jsr*/,
        {T_BYTE}/*ret*/, {}/*tableswitch*/, {}/*lookupswitch*/,
        {}/*ireturn*/, {}/*lreturn*/, {}/*freturn*/, {}/*dreturn*/,
        {}/*areturn*/, {}/*return*/, {T_SHORT}/*getstatic*/,
        {T_SHORT}/*putstatic*/, {T_SHORT}/*getfield*/,
        {T_SHORT}/*putfield*/, {T_SHORT}/*invokevirtual*/,
        {T_SHORT}/*invokespecial*/, {T_SHORT}/*invokestatic*/,
        {T_SHORT, T_BYTE, T_BYTE}/*invokeinterface*/, {},
        {T_SHORT}/*new*/, {T_BYTE}/*newarray*/,
        {T_SHORT}/*anewarray*/, {}/*arraylength*/, {}/*athrow*/,
        {T_SHORT}/*checkcast*/, {T_SHORT}/*instanceof*/,
        {}/*monitorenter*/, {}/*monitorexit*/, {T_BYTE}/*wide*/,
        {T_SHORT, T_BYTE}/*multianewarray*/, {T_SHORT}/*ifnull*/,
        {T_SHORT}/*ifnonnull*/, {T_INT}/*goto_w*/, {T_INT}/*jsr_w*/,
        {}/*breakpoint*/, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}/*impdep1*/, {}/*impdep2*/
    };
    
    /**
     * Names of opcodes.
     */ 
    public static final String[] OPCODE_NAMES = {
        "nop", "aconst_null", "iconst_m1", "iconst_0", "iconst_1",
        "iconst_2", "iconst_3", "iconst_4", "iconst_5", "lconst_0",
        "lconst_1", "fconst_0", "fconst_1", "fconst_2", "dconst_0",
        "dconst_1", "bipush", "sipush", "ldc", "ldc_w", "ldc2_w", "iload",
        "lload", "fload", "dload", "aload", "iload_0", "iload_1", "iload_2",
        "iload_3", "lload_0", "lload_1", "lload_2", "lload_3", "fload_0",
        "fload_1", "fload_2", "fload_3", "dload_0", "dload_1", "dload_2",
        "dload_3", "aload_0", "aload_1", "aload_2", "aload_3", "iaload",
        "laload", "faload", "daload", "aaload", "baload", "caload", "saload",
        "istore", "lstore", "fstore", "dstore", "astore", "istore_0",
        "istore_1", "istore_2", "istore_3", "lstore_0", "lstore_1",
        "lstore_2", "lstore_3", "fstore_0", "fstore_1", "fstore_2",
        "fstore_3", "dstore_0", "dstore_1", "dstore_2", "dstore_3",
        "astore_0", "astore_1", "astore_2", "astore_3", "iastore", "lastore",
        "fastore", "dastore", "aastore", "bastore", "castore", "sastore",
        "pop", "pop2", "dup", "dup_x1", "dup_x2", "dup2", "dup2_x1",
        "dup2_x2", "swap", "iadd", "ladd", "fadd", "dadd", "isub", "lsub",
        "fsub", "dsub", "imul", "lmul", "fmul", "dmul", "idiv", "ldiv",
        "fdiv", "ddiv", "irem", "lrem", "frem", "drem", "ineg", "lneg",
        "fneg", "dneg", "ishl", "lshl", "ishr", "lshr", "iushr", "lushr",
        "iand", "land", "ior", "lor", "ixor", "lxor", "iinc", "i2l", "i2f",
        "i2d", "l2i", "l2f", "l2d", "f2i", "f2l", "f2d", "d2i", "d2l", "d2f",
        "i2b", "i2c", "i2s", "lcmp", "fcmpl", "fcmpg",
        "dcmpl", "dcmpg", "ifeq", "ifne", "iflt", "ifge", "ifgt", "ifle",
        "if_icmpeq", "if_icmpne", "if_icmplt", "if_icmpge", "if_icmpgt",
        "if_icmple", "if_acmpeq", "if_acmpne", "goto", "jsr", "ret",
        "tableswitch", "lookupswitch", "ireturn", "lreturn", "freturn",
        "dreturn", "areturn", "return", "getstatic", "putstatic", "getfield",
        "putfield", "invokevirtual", "invokespecial", "invokestatic",
        "invokeinterface", ILLEGAL_OPCODE, "new", "newarray", "anewarray",
        "arraylength", "athrow", "checkcast", "instanceof", "monitorenter",
        "monitorexit", "wide", "multianewarray", "ifnull", "ifnonnull",
        "goto_w", "jsr_w", "breakpoint", ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE, ILLEGAL_OPCODE,
        ILLEGAL_OPCODE, "impdep1", "impdep2"
    };
    
    /**
     * Number of words consumed on operand stack by instructions.
     */ 
    public static final int[] CONSUME_STACK = {
        0/*nop*/, 0/*aconst_null*/, 0/*iconst_m1*/, 0/*iconst_0*/, 0/*iconst_1*/,
        0/*iconst_2*/, 0/*iconst_3*/, 0/*iconst_4*/, 0/*iconst_5*/, 0/*lconst_0*/,
        0/*lconst_1*/, 0/*fconst_0*/, 0/*fconst_1*/, 0/*fconst_2*/, 0/*dconst_0*/,
        0/*dconst_1*/, 0/*bipush*/, 0/*sipush*/, 0/*ldc*/, 0/*ldc_w*/, 0/*ldc2_w*/, 0/*iload*/,
        0/*lload*/, 0/*fload*/, 0/*dload*/, 0/*aload*/, 0/*iload_0*/, 0/*iload_1*/, 0/*iload_2*/,
        0/*iload_3*/, 0/*lload_0*/, 0/*lload_1*/, 0/*lload_2*/, 0/*lload_3*/, 0/*fload_0*/,
        0/*fload_1*/, 0/*fload_2*/, 0/*fload_3*/, 0/*dload_0*/, 0/*dload_1*/, 0/*dload_2*/,
        0/*dload_3*/, 0/*aload_0*/, 0/*aload_1*/, 0/*aload_2*/, 0/*aload_3*/, 2/*iaload*/,
        2/*laload*/, 2/*faload*/, 2/*daload*/, 2/*aaload*/, 2/*baload*/, 2/*caload*/, 2/*saload*/,
        1/*istore*/, 2/*lstore*/, 1/*fstore*/, 2/*dstore*/, 1/*astore*/, 1/*istore_0*/,
        1/*istore_1*/, 1/*istore_2*/, 1/*istore_3*/, 2/*lstore_0*/, 2/*lstore_1*/,
        2/*lstore_2*/, 2/*lstore_3*/, 1/*fstore_0*/, 1/*fstore_1*/, 1/*fstore_2*/,
        1/*fstore_3*/, 2/*dstore_0*/, 2/*dstore_1*/, 2/*dstore_2*/, 2/*dstore_3*/,
        1/*astore_0*/, 1/*astore_1*/, 1/*astore_2*/, 1/*astore_3*/, 3/*iastore*/, 4/*lastore*/,
        3/*fastore*/, 4/*dastore*/, 3/*aastore*/, 3/*bastore*/, 3/*castore*/, 3/*sastore*/,
        1/*pop*/, 2/*pop2*/, 1/*dup*/, 2/*dup_x1*/, 3/*dup_x2*/, 2/*dup2*/, 3/*dup2_x1*/,
        4/*dup2_x2*/, 2/*swap*/, 2/*iadd*/, 4/*ladd*/, 2/*fadd*/, 4/*dadd*/, 2/*isub*/, 4/*lsub*/,
        2/*fsub*/, 4/*dsub*/, 2/*imul*/, 4/*lmul*/, 2/*fmul*/, 4/*dmul*/, 2/*idiv*/, 4/*ldiv*/,
        2/*fdiv*/, 4/*ddiv*/, 2/*irem*/, 4/*lrem*/, 2/*frem*/, 4/*drem*/, 1/*ineg*/, 2/*lneg*/,
        1/*fneg*/, 2/*dneg*/, 2/*ishl*/, 3/*lshl*/, 2/*ishr*/, 3/*lshr*/, 2/*iushr*/, 3/*lushr*/,
        2/*iand*/, 4/*land*/, 2/*ior*/, 4/*lor*/, 2/*ixor*/, 4/*lxor*/, 0/*iinc*/,
        1/*i2l*/, 1/*i2f*/, 1/*i2d*/, 2/*l2i*/, 2/*l2f*/, 2/*l2d*/, 1/*f2i*/, 1/*f2l*/,
        1/*f2d*/, 2/*d2i*/, 2/*d2l*/, 2/*d2f*/, 1/*i2b*/, 1/*i2c*/, 1/*i2s*/, 
        4/*lcmp*/, 2/*fcmpl*/, 2/*fcmpg*/, 4/*dcmpl*/, 4/*dcmpg*/, 1/*ifeq*/, 1/*ifne*/,
        1/*iflt*/, 1/*ifge*/, 1/*ifgt*/, 1/*ifle*/, 2/*if_icmpeq*/, 2/*if_icmpne*/, 2/*if_icmplt*/,
        2 /*if_icmpge*/, 2/*if_icmpgt*/, 2/*if_icmple*/, 2/*if_acmpeq*/, 2/*if_acmpne*/,
        0/*goto*/, 0/*jsr*/, 0/*ret*/, 1/*tableswitch*/, 1/*lookupswitch*/, 1/*ireturn*/,
        2/*lreturn*/, 1/*freturn*/, 2/*dreturn*/, 1/*areturn*/, 0/*return*/, 0/*getstatic*/,
        UNPREDICTABLE/*putstatic*/, 1/*getfield*/, UNPREDICTABLE/*putfield*/,
        UNPREDICTABLE/*invokevirtual*/, UNPREDICTABLE/*invokespecial*/,
        UNPREDICTABLE/*invokestatic*/,
        UNPREDICTABLE/*invokeinterface*/, UNDEFINED, 0/*new*/, 1/*newarray*/, 1/*anewarray*/,
        1/*arraylength*/, 1/*athrow*/, 1/*checkcast*/, 1/*instanceof*/, 1/*monitorenter*/,
        1/*monitorexit*/, 0/*wide*/, UNPREDICTABLE/*multianewarray*/, 1/*ifnull*/, 1/*ifnonnull*/,
        0/*goto_w*/, 0/*jsr_w*/, 0/*breakpoint*/, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNPREDICTABLE/*impdep1*/, UNPREDICTABLE/*impdep2*/
    };
    
    /**
     * Number of words produced onto operand stack by instructions.
     */ 
    public static final int[] PRODUCE_STACK = {
        0/*nop*/, 1/*aconst_null*/, 1/*iconst_m1*/, 1/*iconst_0*/, 1/*iconst_1*/,
        1/*iconst_2*/, 1/*iconst_3*/, 1/*iconst_4*/, 1/*iconst_5*/, 2/*lconst_0*/,
        2/*lconst_1*/, 1/*fconst_0*/, 1/*fconst_1*/, 1/*fconst_2*/, 2/*dconst_0*/,
        2/*dconst_1*/, 1/*bipush*/, 1/*sipush*/, 1/*ldc*/, 1/*ldc_w*/, 2/*ldc2_w*/, 1/*iload*/,
        2/*lload*/, 1/*fload*/, 2/*dload*/, 1/*aload*/, 1/*iload_0*/, 1/*iload_1*/, 1/*iload_2*/,
        1/*iload_3*/, 2/*lload_0*/, 2/*lload_1*/, 2/*lload_2*/, 2/*lload_3*/, 1/*fload_0*/,
        1/*fload_1*/, 1/*fload_2*/, 1/*fload_3*/, 2/*dload_0*/, 2/*dload_1*/, 2/*dload_2*/,
        2/*dload_3*/, 1/*aload_0*/, 1/*aload_1*/, 1/*aload_2*/, 1/*aload_3*/, 1/*iaload*/,
        2/*laload*/, 1/*faload*/, 2/*daload*/, 1/*aaload*/, 1/*baload*/, 1/*caload*/, 1/*saload*/,
        0/*istore*/, 0/*lstore*/, 0/*fstore*/, 0/*dstore*/, 0/*astore*/, 0/*istore_0*/,
        0/*istore_1*/, 0/*istore_2*/, 0/*istore_3*/, 0/*lstore_0*/, 0/*lstore_1*/,
        0/*lstore_2*/, 0/*lstore_3*/, 0/*fstore_0*/, 0/*fstore_1*/, 0/*fstore_2*/,
        0/*fstore_3*/, 0/*dstore_0*/, 0/*dstore_1*/, 0/*dstore_2*/, 0/*dstore_3*/,
        0/*astore_0*/, 0/*astore_1*/, 0/*astore_2*/, 0/*astore_3*/, 0/*iastore*/, 0/*lastore*/,
        0/*fastore*/, 0/*dastore*/, 0/*aastore*/, 0/*bastore*/, 0/*castore*/, 0/*sastore*/,
        0/*pop*/, 0/*pop2*/, 2/*dup*/, 3/*dup_x1*/, 4/*dup_x2*/, 4/*dup2*/, 5/*dup2_x1*/,
        6/*dup2_x2*/, 2/*swap*/, 1/*iadd*/, 2/*ladd*/, 1/*fadd*/, 2/*dadd*/, 1/*isub*/, 2/*lsub*/,
        1/*fsub*/, 2/*dsub*/, 1/*imul*/, 2/*lmul*/, 1/*fmul*/, 2/*dmul*/, 1/*idiv*/, 2/*ldiv*/,
        1/*fdiv*/, 2/*ddiv*/, 1/*irem*/, 2/*lrem*/, 1/*frem*/, 2/*drem*/, 1/*ineg*/, 2/*lneg*/,
        1/*fneg*/, 2/*dneg*/, 1/*ishl*/, 2/*lshl*/, 1/*ishr*/, 2/*lshr*/, 1/*iushr*/, 2/*lushr*/,
        1/*iand*/, 2/*land*/, 1/*ior*/, 2/*lor*/, 1/*ixor*/, 2/*lxor*/,
        0/*iinc*/, 2/*i2l*/, 1/*i2f*/, 2/*i2d*/, 1/*l2i*/, 1/*l2f*/, 2/*l2d*/, 1/*f2i*/,
        2/*f2l*/, 2/*f2d*/, 1/*d2i*/, 2/*d2l*/, 1/*d2f*/,
        1/*i2b*/, 1/*i2c*/, 1/*i2s*/, 1/*lcmp*/, 1/*fcmpl*/, 1/*fcmpg*/,
        1/*dcmpl*/, 1/*dcmpg*/, 0/*ifeq*/, 0/*ifne*/, 0/*iflt*/, 0/*ifge*/, 0/*ifgt*/, 0/*ifle*/,
        0/*if_icmpeq*/, 0/*if_icmpne*/, 0/*if_icmplt*/, 0/*if_icmpge*/, 0/*if_icmpgt*/,
        0/*if_icmple*/, 0/*if_acmpeq*/, 0/*if_acmpne*/, 0/*goto*/, 1/*jsr*/, 0/*ret*/,
        0/*tableswitch*/, 0/*lookupswitch*/, 0/*ireturn*/, 0/*lreturn*/, 0/*freturn*/,
        0/*dreturn*/, 0/*areturn*/, 0/*return*/, UNPREDICTABLE/*getstatic*/, 0/*putstatic*/,
        UNPREDICTABLE/*getfield*/, 0/*putfield*/, UNPREDICTABLE/*invokevirtual*/,
        UNPREDICTABLE/*invokespecial*/, UNPREDICTABLE/*invokestatic*/,
        UNPREDICTABLE/*invokeinterface*/, UNDEFINED, 1/*new*/, 1/*newarray*/, 1/*anewarray*/,
        1/*arraylength*/, 1/*athrow*/, 1/*checkcast*/, 1/*instanceof*/, 0/*monitorenter*/,
        0/*monitorexit*/, 0/*wide*/, 1/*multianewarray*/, 0/*ifnull*/, 0/*ifnonnull*/,
        0/*goto_w*/, 1/*jsr_w*/, 0/*breakpoint*/, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        UNDEFINED, UNPREDICTABLE/*impdep1*/, UNPREDICTABLE/*impdep2*/
    };
    
}
