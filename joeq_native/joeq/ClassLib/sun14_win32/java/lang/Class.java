package ClassLib.sun14_win32.java.lang;

public final class Class {
    
    private native java.lang.reflect.Field[] getFields0(int m);
    private native java.lang.reflect.Method[] getMethods0(int m);
    private native java.lang.reflect.Constructor[] getConstructors0(int m);
    
    private java.lang.reflect.Field[] getDeclaredFields0(boolean publicOnly) {
	java.lang.reflect.Field[] f = getFields0(java.lang.reflect.Member.DECLARED);
	if (publicOnly) {
	    int count = 0;
	    for (int i=0; i<f.length; ++i) {
		if (java.lang.reflect.Modifier.isPublic(f[i].getModifiers()))
		    ++count;
	    }
	    java.lang.reflect.Field[] f2 = new java.lang.reflect.Field[count];
	    --count;
	    for (int i=0, j=-1; j<count; ++i) {
		if (java.lang.reflect.Modifier.isPublic(f[i].getModifiers()))
		    f2[++j] = f[i];
	    }
	    f = f2;
	}
	return f;
    }
    private java.lang.reflect.Method[] getDeclaredMethods0(boolean publicOnly) {
	java.lang.reflect.Method[] f = getMethods0(java.lang.reflect.Member.DECLARED);
	if (publicOnly) {
	}
	return f;
    }
    private java.lang.reflect.Constructor[] getDeclaredConstructors0(boolean publicOnly) {
	java.lang.reflect.Constructor[] f = getConstructors0(java.lang.reflect.Member.DECLARED);
	if (publicOnly) {
	}
	return f;
    }
    
    private static boolean desiredAssertionStatus0(Class clazz) {
	// TODO.
	return false;
    }
}
