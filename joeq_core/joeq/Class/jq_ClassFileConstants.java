/*
 * jq_ClassFileConstants.java
 *
 * Created on December 19, 2000, 9:02 AM
 *
 * @author  jwhaley
 * @version 
 */

package Clazz;

public interface jq_ClassFileConstants {

    public static final char ACC_PUBLIC       = 0x0001;
    public static final char ACC_PRIVATE      = 0x0002;
    public static final char ACC_PROTECTED    = 0x0004;
    public static final char ACC_STATIC       = 0x0008;
    public static final char ACC_FINAL        = 0x0010;
    public static final char ACC_SYNCHRONIZED = 0x0020; // same value
    public static final char ACC_SUPER        = 0x0020; // same value
    public static final char ACC_VOLATILE     = 0x0040;
    public static final char ACC_TRANSIENT    = 0x0080;
    public static final char ACC_NATIVE       = 0x0100;
    public static final char ACC_INTERFACE    = 0x0200;
    public static final char ACC_ABSTRACT     = 0x0400;
    public static final char ACC_STRICT       = 0x0800;

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
    
    public static final byte TC_BYTE     = (byte)'B';
    public static final byte TC_CHAR     = (byte)'C';
    public static final byte TC_DOUBLE   = (byte)'D';
    public static final byte TC_FLOAT    = (byte)'F';
    public static final byte TC_INT      = (byte)'I';
    public static final byte TC_LONG     = (byte)'J';
    public static final byte TC_CLASS    = (byte)'L';
    public static final byte TC_CLASSEND = (byte)';';
    public static final byte TC_SHORT    = (byte)'S';
    public static final byte TC_BOOLEAN  = (byte)'Z';
    public static final byte TC_ARRAY    = (byte)'[';
    public static final byte TC_PARAM    = (byte)'(';
    public static final byte TC_PARAMEND = (byte)')';
    public static final byte TC_VOID     = (byte)'V';

    public static final byte T_BOOLEAN = 4;
    public static final byte T_CHAR    = 5;
    public static final byte T_FLOAT   = 6;
    public static final byte T_DOUBLE  = 7;
    public static final byte T_BYTE    = 8;
    public static final byte T_SHORT   = 9;
    public static final byte T_INT     = 10;
    public static final byte T_LONG    = 11;

    public static final int BOOTSTRAP_CHAR_INDEX = 3;
    public static final byte BOOTSTRAP_FROM_CHAR = (byte)'_';
    public static final byte BOOTSTRAP_TO_CHAR = (byte)'a';
    
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
    // This class has been loaded and all members have been created.
    public static final byte STATE_LOADED       = 3;
    // A thread is in the process of verifying this class. (Jvm spec 2.17.3)
    // (see verify pass 2 Jvm spec 4.9.1)
    // It checks the code in each declared method in the class.
    public static final byte STATE_VERIFYING    = 4;
    // This class has been successfully verified.
    public static final byte STATE_VERIFIED     = 5;
    // A thread is in the process of preparing this class. (Jvm spec 2.17.3)
    // Preparation lays out the object fields and creates a method table.
    // Static fields are created and initialized in the NEXT step.
    public static final byte STATE_PREPARING    = 6;
    // This class has been prepared.
    public static final byte STATE_PREPARED     = 7;
    // A thread is creating the static fields for the class, and initializing the
    // ones that have ConstantValue attributes.
    public static final byte STATE_SFINITIALIZING = 8;
    // This class has its static fields created and initialized.
    public static final byte STATE_SFINITIALIZED = 9;
    // A thread is in the process of initializing this class. (Jvm spec 2.17.4-5)
    // Initialization is triggered when code is about to execute that will create
    // an instance, execute a static method, or use or assign a nonconstant static
    // field.
    public static final byte STATE_CLSINITIALIZING = 10;
    public static final byte STATE_CLSINITRUNNING = 11;
    // An error occurred during initialization!  This resulted in a throwing of 
    // a NoClassDefFoundError, ExceptionInInitializerError, or OutOfMemoryError
    // for the initializing thread.  Any further attempts to initialize should
    // result in the throwing of a NoClassDefFoundError. 
    public static final byte STATE_CLSINITERROR = 12;
    // This class has been fully initialized!
    public static final byte STATE_CLSINITIALIZED  = 13;

}
