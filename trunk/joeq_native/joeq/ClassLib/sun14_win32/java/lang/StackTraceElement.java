package ClassLib.sun14_win32.java.lang;

public final class StackTraceElement {
    private java.lang.String declaringClass;
    private java.lang.String methodName;
    private java.lang.String fileName;
    private int lineNumber;

    StackTraceElement(java.lang.String declaringClass,
		      java.lang.String methodName,
		      java.lang.String fileName,
		      int lineNumber) {
	this.declaringClass = declaringClass;
	this.methodName = methodName;
	this.fileName = fileName;
	this.lineNumber = lineNumber;
    }
}
