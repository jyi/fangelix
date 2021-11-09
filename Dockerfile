FROM ubuntu:18.04


MAINTAINER Jooyong Yi <jooyong@unsit.ac.kr>

RUN apt -y -q update
RUN apt -y -q upgrade

# Other dependencies
RUN apt -y install xz-utils
RUN apt -y install build-essential
RUN apt -y install subversion
RUN apt -y install bison
RUN apt -y install flex
RUN apt -y install libcap-dev
RUN apt -y install cmake
RUN apt -y install libncurses5-dev
RUN apt -y install libboost-all-dev
RUN apt -y install wget
RUN apt -y install lsb-core
RUN apt -y install silversearcher-ag
RUN apt -y install global
RUN apt -y install python3-pip
RUN apt -y install python-pip
RUN apt -y install libblas-dev
RUN apt -y install maven
RUN apt -y install gdb
RUN apt -y install git autoconf automake libtool
RUN apt -y install libsqlite3-dev
RUN apt -y install libcap-dev
RUN apt -y install libacl1-dev
RUN apt-get install software-properties-common -y --no-install-recommends

# Install Java
RUN apt-get purge icedtea-* openjdk-* -y
RUN add-apt-repository -y ppa:openjdk-r/ppa && apt-get update && apt-get install -y openjdk-8-jdk
#check if java command is pointing to " link currently points to /opt/jdk/jdk1.8.0_05/bin/java"
RUN update-alternatives --display java

#check if java command is pointing to " link currently points to /opt/jdk/jdk1.8.0_05/bin/javac"
RUN update-alternatives --display javac

RUN java -version
RUN javac -version


# Install dejagnu
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt -y install dejagnu

# Install sbt
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
RUN apt-get update
RUN apt -y install sbt

# Installing FAngelix
RUN bash -c 'git clone --recursive https://github.com/jyi/fangelix.git'

WORKDIR fangelix
RUN bash -c 'source activate && make llvm-gcc'
RUN bash -c 'source activate && make llvm2'
RUN bash -c 'source activate && make minisat'
RUN bash -c 'source activate && make stp'
RUN bash -c 'source activate && make klee-uclibc'
RUN bash -c 'source activate && make klee'
RUN bash -c 'source activate && make z3'
RUN bash -c 'source activate && make clang'
RUN bash -c 'source activate && make bear'
RUN bash -c 'source activate && make runtime'
RUN bash -c 'source activate && make frontend'
RUN bash -c 'source activate && make maxsmt'
RUN bash -c 'source activate && make synthesis'
ENV VER=3.6.3
RUN wget http://www-eu.apache.org/dist/maven/maven-3/${VER}/binaries/apache-maven-${VER}-bin.tar.gz
RUN bash -c 'tar xvf apache-maven-${VER}-bin.tar.gz'
RUN bash -c 'rm apache-maven-${VER}-bin.tar.gz'
RUN bash -c 'mv apache-maven-${VER} /opt/maven'
RUN bash -c 'echo "export MAVEN_HOME=/opt/maven;export PATH=\$MAVEN_HOME/bin:\$PATH:\$MAVEN_HOME/bin" > /etc/profile.d/maven.sh'
RUN bash -c 'source /etc/profile.d/maven.sh && mvn -version'
RUN bash -c 'source activate && source /etc/profile.d/maven.sh && make nsynth'

ENV LC_CTYPE C.UTF-8
ENV LC_ALL C.UTF-8
RUN pip3 install watchdog
WORKDIR /fangelix
RUN rm build/*.xz build/*.tgz build/*.bz2

# Installing other dependencies
RUN bash -c 'pip3 install numpy scipy theano pandas tqdm h5py'
RUN apt -y install ksh
RUN bash -c 'cpan Text::CSV Text::Trim'
