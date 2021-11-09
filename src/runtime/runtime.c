#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <dirent.h>
#include <assert.h>
#include "klee/klee.h"
#include "runtime.h"
#include "cJSON.h"


typedef struct _arr_values {
  int size;
  int *values;
} arr_values;

typedef arr_values *arr_values_ptr;

/*
  Hashtable implementation (for positive integers)
*/

int table_miss = 1;

#define ENTRY_S(type)                           \
struct entry_s_##type {                         \
  char *key;                                    \
  type value;                                   \
  struct entry_s_##type *next;                  \
};

ENTRY_S(int)
ENTRY_S(arr_values_ptr)
#undef ENTRY_S


#define ENTRY_T(type)                           \
typedef struct entry_s_##type entry_t_##type;

ENTRY_T(int)
ENTRY_T(arr_values_ptr)
#undef ENTRY_T


#define HASHTABLE_S(type)                               \
struct hashtable_s_##type {                             \
  int size;                                             \
  struct entry_s_##type **table;                        \
};                                                      \
                                                        \
typedef struct hashtable_s_##type hashtable_t_##type;

HASHTABLE_S(int)
HASHTABLE_S(arr_values_ptr)
#undef HASHTABLE_S


#define HASHTABLE_CREATE(type)                                          \
  /* Create a new hashtable. */                                         \
  hashtable_t_##type * type##_ht_create( int size ) {                   \
    hashtable_t_##type *hashtable = NULL;                               \
    int i;                                                              \
                                                                        \
    if( size < 1 ) return NULL;                                         \
                                                                        \
    /* Allocate the table itself. */                                    \
    if( ( hashtable = (hashtable_t_##type *) malloc( sizeof( hashtable_t_##type ) ) ) == NULL ) { \
      return NULL;                                                      \
    }                                                                   \
                                                                        \
    /* Allocate pointers to the head nodes. */                          \
    if( ( hashtable->table = (entry_t_##type **) malloc( sizeof( entry_t_##type * ) * size ) ) == NULL ) { \
      return NULL;                                                      \
    }                                                                   \
    for( i = 0; i < size; i++ ) {                                       \
      hashtable->table[i] = NULL;                                       \
    }                                                                   \
                                                                        \
    hashtable->size = size;                                             \
                                                                        \
    return hashtable;                                                   \
}

HASHTABLE_CREATE(int)
HASHTABLE_CREATE(arr_values_ptr)
#undef HASHTABLE_CREATE


#define HASHTABLE_NEWPAIR(type)                                         \
  /* Create a key-value pair. */                                        \
  entry_t_##type * type##_ht_newpair( char *key, type value ) {         \
    entry_t_##type *newpair;                                            \
                                                                        \
    if( ( newpair = (entry_t_##type *) malloc( sizeof( entry_t_##type ) ) ) == NULL ) { \
      return NULL;                                                      \
    }                                                                   \
                                                                        \
    if( ( newpair->key = strdup( key ) ) == NULL ) {                    \
      return NULL;                                                      \
    }                                                                   \
                                                                        \
    newpair->value = value;                                             \
    newpair->next = NULL;                                               \
    return newpair;                                                     \
}

HASHTABLE_NEWPAIR(int)
HASHTABLE_NEWPAIR(arr_values_ptr)
#undef HASHTABLE_NEWPAIR


#define HASHTABLE_SET(type)                                             \
  /* Insert a key-value pair into a hash table. */                      \
  void type##_ht_set(hashtable_t_##type *hashtable, char *key,          \
                     type value) {                                      \
    int bin = 0;                                                        \
    entry_t_##type *newpair = NULL;                                     \
    entry_t_##type *next = NULL;                                        \
    entry_t_##type *last = NULL;                                        \
                                                                        \
    bin = type##_ht_hash( hashtable, key );                             \
    next = hashtable->table[ bin ];                                     \
                                                                        \
    while( next != NULL && next->key != NULL &&                         \
           strcmp( key, next->key ) > 0 ) {                             \
      last = next;                                                      \
      next = next->next;                                                \
    }                                                                   \
                                                                        \
    /* There's already a pair.  Let's replace that string. */           \
    if( next != NULL && next->key != NULL &&                            \
        strcmp( key, next->key ) == 0 ) {                               \
      next->value = value;                                              \
      /* Nope, could't find it.  Time to grow a pair. */                \
    } else {                                                            \
      newpair = type##_ht_newpair( key, value );                        \
                                                                        \
      /* We're at the start of the linked list in this bin. */          \
      if( next == hashtable->table[ bin ] ) {                           \
        newpair->next = next;                                           \
        hashtable->table[ bin ] = newpair;                              \
                                                                        \
        /* We're at the end of the linked list in this bin. */          \
      } else if ( next == NULL ) {                                      \
        last->next = newpair;                                           \
                                                                        \
        /* We're in the middle of the list. */                          \
      } else  {                                                         \
        newpair->next = next;                                           \
        last->next = newpair;                                           \
      }                                                                 \
    }                                                                   \
}

HASHTABLE_SET(int)
HASHTABLE_SET(arr_values_ptr)
#undef HASHTABLE_SET

#define HASHTABLE_GET(type)                                             \
  /* Retrieve a key-value pair from a hash table. */                    \
  type type##_ht_get( hashtable_t_##type *hashtable, char *key ) {      \
    int bin = 0;                                                        \
    entry_t_##type *pair;                                               \
                                                                        \
    bin = type##_ht_hash( hashtable, key );                             \
                                                                        \
    /* Step through the bin, looking for our value. */                  \
    pair = hashtable->table[ bin ];                                     \
    while( pair != NULL && pair->key != NULL && strcmp( key, pair->key ) > 0 ) { \
      pair = pair->next;                                                \
    }                                                                   \
                                                                        \
    /* Did we actually find anything? */                                \
    if( pair == NULL || pair->key == NULL || strcmp( key, pair->key ) != 0 ) { \
      table_miss = 1;                                                   \
      return 0;                                                         \
                                                                        \
    } else {                                                            \
      table_miss = 0;                                                   \
      return pair->value;                                               \
    }                                                                   \
  }

HASHTABLE_GET(int)
HASHTABLE_GET(arr_values_ptr)
#undef HASHTABLE_GET


#define HASHTABLE_HAS(type)                                             \
  bool type##_ht_has( hashtable_t_##type *hashtable, char *key ) {      \
    type##_ht_get(hashtable, key);                                      \
    if (table_miss) return false;                                       \
    else return true;                                                   \
  }

HASHTABLE_HAS(int)
HASHTABLE_HAS(arr_values_ptr)
#undef HASHTABLE_HAS


#define HASHTABLE_HASH(type)                                            \
/* Hash a string for a particular hash table. */                        \
int type##_ht_hash( hashtable_t_##type *hashtable, char *key ) {        \
  unsigned long int hashval;                                            \
  int i = 0;                                                            \
                                                                        \
  /* Convert our string to an integer */                                \
  while( hashval < ULONG_MAX && i < strlen( key ) ) {                   \
    hashval = hashval << 8;                                             \
    hashval += key[ i ];                                                \
    i++;                                                                \
  }                                                                     \
                                                                        \
  return hashval % hashtable->size;                                     \
}

HASHTABLE_HASH(int)
HASHTABLE_HASH(arr_values_ptr)
#undef HASHTABLE_HASH


/*
  End of hashtable implementation
*/

#define TABLE_SIZE 65536
#define MAX_PATH_LENGTH 1000
#define MAX_NAME_LENGTH 1000
#define INT_LENGTH 15
#define LONG_LENGTH (INT_LENGTH * 2)

hashtable_t_int *tab_output_instances = NULL;
hashtable_t_int *tab_choice_instances = NULL;
hashtable_t_int *tab_const_choices = NULL;

hashtable_t_int *set_ids;

void init_tables() {
  tab_output_instances = int_ht_create(TABLE_SIZE);
  tab_choice_instances = int_ht_create(TABLE_SIZE);
  tab_const_choices = int_ht_create(TABLE_SIZE);
}

/*
  JSON
*/

cJSON *load_json = NULL;
hashtable_t_arr_values_ptr *tab_arr_values_ptr;

void make_load_json_ready() {
  FILE *f = NULL;
  long len = 0;
  char *data = NULL;
  char *out = NULL;
  const cJSON *values_cjson = NULL;
  int idx = 0;

  /* delete the existing one */
  if (load_json) {
    cJSON_Delete(load_json);
    load_json = NULL;
  }

  /* open in read binary mode */
  f = fopen(getenv("ANGELIX_LOAD_JSON"), "rb");
  /* get the length */
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  fseek(f, 0, SEEK_SET);

  data = (char*)malloc(len + 1);

  fread(data, 1, len, f);
  data[len] = '\0';
  fclose(f);

  /* parase the json text */
  load_json = cJSON_Parse(data);
  if (!load_json) {
    fprintf(stderr, "[runtime] Error before: [%s]\n", cJSON_GetErrorPtr());
    exit(1);
  }

  tab_arr_values_ptr = arr_values_ptr_ht_create(TABLE_SIZE);
}

void init_set_ids() {
  if (!load_json) {
    fprintf(stderr, "[runtime] Error before: [%s]\n", cJSON_GetErrorPtr());
    exit(1);
  }

  set_ids = int_ht_create(TABLE_SIZE);
  cJSON *loc = NULL;
  cJSON_ArrayForEach(loc, load_json) {
    char *id = loc->string;
    if (id != NULL) {
      int_ht_set(set_ids, id, 1);
    }
  }
}

int load_instance_json(char* loc, int instance) {
  const cJSON *bit_vector = NULL;
  const cJSON *bit = NULL;
  arr_values_ptr avp = NULL;
  int *values = NULL;
  int idx = 0;

  if (!load_json) {
    fprintf(stderr, "[runtime] Error before: [%s]\n",
            cJSON_GetErrorPtr());
    exit(1);
  }

  avp = arr_values_ptr_ht_get(tab_arr_values_ptr, loc);
  if (table_miss) {
    bit_vector = cJSON_GetObjectItemCaseSensitive(load_json, loc);
    avp = (arr_values_ptr) malloc(sizeof(arr_values));
    avp->size = cJSON_GetArraySize(bit_vector);
    values = (int*) malloc(sizeof(int) * avp->size);
    cJSON_ArrayForEach(bit, bit_vector) {
      values[idx++] = bit->valueint;
    }
    avp->values = values;
    arr_values_ptr_ht_set(tab_arr_values_ptr, loc, avp);
  }

  if (instance < avp->size) {
    return avp->values[instance];
  } else {
    /* return rand() % 2; */
    /* if not specified, return 0 by default.*/
    /* Rationale: avoid infinite loop */
    return 0;
  }
}

/*
  Parsing and printing
*/

int parse_int(char* str) {
  return atoi(str);
}

bool parse_bool(char* str) {
  if (strncmp(str, "true", 4) == 0) {
    return true;
  }
  if (strncmp(str, "false", 5) == 0) {
    return false;
  }
  fprintf(stderr, "[runtime] wrong boolean format: %s\n", str);
  abort();
}

char parse_char(char* str) {
  if (strlen(str) != 1) {
    fprintf(stderr, "[runtime] wrong character format: %s\n", str);
    abort();
  }
  return str[0];
}

char* print_int(int i) {
  char* str = (char*) malloc(INT_LENGTH * sizeof(char));
  sprintf(str, "%d", i);
  return str;
}

char* print_long(long i) {
  char* str = (char*) malloc(LONG_LENGTH * sizeof(char));
  sprintf(str, "%ld", i);
  return str;
}

char* print_bool(bool b) {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

char* print_char(char c) {
  char* str = (char*) malloc(2 * sizeof(char));
  str[1] = '\0';
  str[0] = c;
  return str;
}

char* print_str(char* s) {
  return s;
}

/*
  Loading and dumping
*/

char* load_instance(char* var, int instance) {
  char *dir = getenv("ANGELIX_LOAD");
  char file[MAX_PATH_LENGTH + 1];
  sprintf(file, "%s/%s/%d", dir, var, instance);

  FILE *fp = fopen(file, "r");
  if (fp == NULL)
    return NULL;

  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *string = (char*) malloc(fsize + 1);
  fread(string, fsize, 1, fp);
  fclose(fp);

  string[fsize] = 0;
  return string;
}

void dump_instance(char* var, int instance, char* data) {
  char *dir = getenv("ANGELIX_DUMP");
  char vardir[MAX_PATH_LENGTH + 1];
  sprintf(vardir, "%s/%s", dir, var, instance);

  DIR* d = opendir(vardir);
  if (d) {
    closedir(d);
  } else {
    if (mkdir(vardir, 0777) == -1) {
      fprintf(stderr, "[runtime] failed to make %s\n", vardir);
      exit(1);
    }
  }

  char file[MAX_PATH_LENGTH + 1];
  sprintf(file, "%s/%d", vardir, instance);

  FILE *fp = fopen(file, "w");
  if (!fp)
    abort();
  fputs(data, fp);
  fclose(fp);
}

/*
  Reading dumped data
*/

#define LOOKUP_RESULT_PROTO(type)                               \
  struct type##_lookup_result { bool succeed; type value; };

LOOKUP_RESULT_PROTO(int)
LOOKUP_RESULT_PROTO(bool)
LOOKUP_RESULT_PROTO(char)

#undef LOOKUP_RESULT_PROTO

#define LOAD_PROTO(type)                                                \
  struct type##_lookup_result load_##type(char* var, int instance) {    \
    struct type##_lookup_result result;                                 \
    result.succeed = false;                                             \
    result.value = 0;                                                   \
                                                                        \
    char* data = load_instance(var, instance);                          \
                                                                        \
    if (data != NULL) {                                                 \
      result.succeed = true;                                            \
      result.value = parse_##type(data);                                \
    }                                                                   \
                                                                        \
    return result;                                                      \
  }

LOAD_PROTO(int)
LOAD_PROTO(bool)
LOAD_PROTO(char)

#undef LOAD_PROTO

#define DUMP_PROTO(type)                                        \
  void dump_##type(char* var, int instance, type value) {       \
    dump_instance(var, instance, print_##type(value));          \
  }

DUMP_PROTO(int)
DUMP_PROTO(long)
DUMP_PROTO(bool)
DUMP_PROTO(char)
DUMP_PROTO(str)

#undef WRITE_TO_FILE_PROTO

#define SYMBOLIC_OUTPUT_PROTO(type, typestr)                    \
  type angelix_symbolic_output_##type(type expr, char* id) {    \
    if (!tab_output_instances)                                  \
      init_tables();                                            \
    int previous = int_ht_get(tab_output_instances, id);        \
    int instance;                                               \
    if (table_miss) {                                           \
      instance = 0;                                             \
    } else {                                                    \
      instance = previous + 1;                                  \
    }                                                           \
    int_ht_set(tab_output_instances, id, instance);             \
    char name[MAX_NAME_LENGTH];                                 \
    sprintf(name, "%s!output!%s!%d", typestr, id, instance);    \
    type s;                                                     \
    klee_make_symbolic(&s, sizeof(s), name);                    \
    klee_assume(s == expr);                                     \
    return s;                                                   \
  }

SYMBOLIC_OUTPUT_PROTO(int, "int")
SYMBOLIC_OUTPUT_PROTO(long, "long")
SYMBOLIC_OUTPUT_PROTO(bool, "bool")
SYMBOLIC_OUTPUT_PROTO(char, "char")

#undef SYMBOLIC_OUTPUT_PROTO


//TODO: later I need to express it through angelix_symbolic_output_str
void angelix_symbolic_reachable(char* id) {
  if (!tab_output_instances)
    init_tables();
  int previous = int_ht_get(tab_output_instances, "reachable");
  int instance;
  if (table_miss) {
    instance = 0;
  } else {
    instance = previous + 1;
  }
  int_ht_set(tab_output_instances, "reachable", instance);
  char name[MAX_NAME_LENGTH];
  sprintf(name, "reachable!%s!%d", id, instance);
  int s;
  klee_make_symbolic(&s, sizeof(int), name);
  klee_assume(s);
}


#define DUMP_OUTPUT_PROTO(type)                                 \
  type angelix_dump_output_##type(type expr, char* id) {        \
    if (getenv("ANGELIX_DUMP")) {                               \
      if (!tab_output_instances)                                \
        init_tables();                                          \
      int previous = int_ht_get(tab_output_instances, id);      \
      int instance;                                             \
      if (table_miss) {                                         \
        instance = 0;                                           \
      } else {                                                  \
        instance = previous + 1;                                \
      }                                                         \
      int_ht_set(tab_output_instances, id, instance);           \
      dump_##type(id, instance, expr);                          \
      return expr;                                              \
    } else {                                                    \
      return expr;                                              \
    }                                                           \
  }

DUMP_OUTPUT_PROTO(int)
DUMP_OUTPUT_PROTO(long)
DUMP_OUTPUT_PROTO(bool)
DUMP_OUTPUT_PROTO(char)

#undef DUMP_OUTPUT_PROTO


//TODO: later I need to express it through angelix_dump_output_str
void angelix_dump_reachable(char* id) {
  if (getenv("ANGELIX_DUMP")) {
    if (!tab_output_instances)
      init_tables();
    int previous = int_ht_get(tab_output_instances, "reachable");
    int instance;
    if (table_miss) {
      instance = 0;
    } else {
      instance = previous + 1;
    }
    int_ht_set(tab_output_instances, "reachable", instance);
    dump_str("reachable", instance, id);
  }
  return;
}


#define CHOOSE_WITH_DEPS_PROTO(type, typestr)                           \
  int angelix_choose_##type##_with_deps(int expr,                       \
                                        int bl, int bc, int el, int ec, \
                                        char** env_ids,                 \
                                        int* env_vals,                  \
                                        int env_size) {                 \
    if (!tab_choice_instances)                                          \
      init_tables();                                                    \
    char str_id[INT_LENGTH * 4 + 4 + 1];                                \
    sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);                     \
    int previous = int_ht_get(tab_choice_instances, str_id);            \
    int instance;                                                       \
    if (table_miss) {                                                   \
      instance = 0;                                                     \
    } else {                                                            \
      instance = previous + 1;                                          \
    }                                                                   \
    int_ht_set(tab_choice_instances, str_id, instance);                 \
    int i;                                                              \
    for (i = 0; i < env_size; i++) {                                    \
      char name[MAX_NAME_LENGTH];                                       \
      char* env_fmt = "int!choice!%d!%d!%d!%d!%d!env!%s";               \
      sprintf(name, env_fmt, bl, bc, el, ec, instance, env_ids[i]);     \
      int sv;                                                           \
      klee_make_symbolic(&sv, sizeof(sv), name);                        \
      klee_assume(sv == env_vals[i]);                                   \
    }                                                                   \
                                                                        \
    char name_orig[MAX_NAME_LENGTH];                                    \
    char* orig_fmt = "%s!choice!%d!%d!%d!%d!%d!original";               \
    sprintf(name_orig, orig_fmt, typestr, bl, bc, el, ec, instance);    \
    int so;                                                             \
    klee_make_symbolic(&so, sizeof(so), name_orig);                     \
    klee_assume(so == expr);                                            \
                                                                        \
    char name[MAX_NAME_LENGTH];                                         \
    char* angelic_fmt = "%s!choice!%d!%d!%d!%d!%d!angelic";             \
    sprintf(name, angelic_fmt, typestr, bl, bc, el, ec, instance);      \
    int s;                                                              \
    klee_make_symbolic(&s, sizeof(s), name);                            \
                                                                        \
    return s;                                                           \
  }

CHOOSE_WITH_DEPS_PROTO(int, "int")
CHOOSE_WITH_DEPS_PROTO(bool, "bool")

#undef CHOOSE_WITH_DEPS_PROTO

#define CHOOSE_PROTO(type, env_val_type, typename, typestr)             \
  type angelix_choose_##typename(int bl, int bc, int el, int ec,        \
                                 char** env_ids,                        \
                                 env_val_type* env_vals,                \
                                 int env_size) {                        \
  if (!tab_choice_instances)                                            \
    init_tables();                                                      \
  char str_id[INT_LENGTH * 4 + 4 + 1];                                  \
  sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);                       \
  int previous = int_ht_get(tab_choice_instances, str_id);              \
  int instance;                                                         \
  if (table_miss) {                                                     \
    instance = 0;                                                       \
  } else {                                                              \
    instance = previous + 1;                                            \
  }                                                                     \
  int_ht_set(tab_choice_instances, str_id, instance);                   \
  int i;                                                                \
  for (i = 0; i < env_size; i++) {                                      \
    char name[MAX_NAME_LENGTH];                                         \
    char* env_fmt = "int!choice!%d!%d!%d!%d!%d!env!%s";                 \
    sprintf(name, env_fmt, bl, bc, el, ec, instance, env_ids[i]);       \
    int sv;                                                             \
    klee_make_symbolic(&sv, sizeof(sv), name);                          \
    klee_assume(sv == env_vals[i]);                                     \
  }                                                                     \
                                                                        \
  char name[MAX_NAME_LENGTH];                                           \
  char* angelic_fmt = "%s!choice!%d!%d!%d!%d!%d!angelic";               \
  sprintf(name, angelic_fmt, typestr, bl, bc, el, ec, instance);        \
  type s;                                                               \
  klee_make_symbolic(&s, sizeof(s), name);                              \
                                                                        \
  return s;                                                             \
}

CHOOSE_PROTO(int, int, int, "int")
CHOOSE_PROTO(bool, int, bool, "bool")
// CHOOSE_PROTO(void*, void*, void_ptr, "void_ptr")

#undef CHOOSE_PROTO

void* angelix_choose_void_ptr(int bl, int bc, int el, int ec,
                              char** env_ids,
                              void** env_vals,
                              int env_size) {
  if (!tab_choice_instances)
    init_tables();
  char str_id[INT_LENGTH * 4 + 4 + 1];
  sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);
  int previous = int_ht_get(tab_choice_instances, str_id);
  int instance;
  if (table_miss) {
    instance = 0;
  } else {
    instance = previous + 1;
  }
  int_ht_set(tab_choice_instances, str_id, instance);
  int i;
  for (i = 0; i < env_size; i++) {
    char name[MAX_NAME_LENGTH];
    char* env_fmt = "int!choice!%d!%d!%d!%d!%d!env!%s";
    sprintf(name, env_fmt, bl, bc, el, ec, instance, env_ids[i]);
    int sv;
    klee_make_symbolic(&sv, sizeof(sv), name);
    klee_assume(sv == (int) env_vals[i]);
  }

  char name[MAX_NAME_LENGTH];
  char* angelic_fmt = "int!choice!%d!%d!%d!%d!%d!angelic";
  sprintf(name, angelic_fmt, bl, bc, el, ec, instance);
  void* s;
  klee_make_symbolic(&s, sizeof(s), name);

  int s_ptr_idx;
  klee_make_symbolic(&s_ptr_idx, sizeof(s_ptr_idx), "ptr_idx");
  klee_assume(s_ptr_idx >= 0);
  klee_assume(s_ptr_idx < env_size);
  int c_idx;
  for (c_idx = 0; c_idx < env_size; c_idx++) {
    if (s_ptr_idx == c_idx) break;
  }
  klee_assume(s == env_vals[s_ptr_idx]);

  return env_vals[s_ptr_idx];
}

#define CHOOSE_CONST_PROTO(type, typestr)                               \
  int angelix_choose_const_##type(int bl, int bc, int el, int ec) {     \
    if (!tab_const_choices)                                             \
      init_tables();                                                    \
    char str_id[INT_LENGTH * 4 + 4 + 1];                                \
    sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);                     \
    int choice = int_ht_get(tab_const_choices, str_id);                 \
    if (table_miss) {                                                   \
      char name[MAX_NAME_LENGTH];                                       \
      char* angelic_fmt = "%s!const!%d!%d!%d!%d";                       \
      sprintf(name, angelic_fmt, typestr, bl, bc, el, ec);              \
      int s;                                                            \
      klee_make_symbolic(&s, sizeof(s), name);                          \
      int_ht_set(tab_const_choices, str_id, s);                         \
      return s;                                                         \
    } else {                                                            \
      return choice;                                                    \
    }                                                                   \
  }

CHOOSE_CONST_PROTO(int, "int")
CHOOSE_CONST_PROTO(bool, "bool")

#undef CHOOSE_CONST_PROTO

#define CHOOSE_WITH_GUIDE_PROTO(type, typestr)                          \
  int angelix_choose_##type_with_guide(int bl, int bc, int el, int ec,  \
                                       char** env_ids,                  \
                                       int* env_vals,                   \
                                       int env_size) {                  \
    if (!tab_choice_instances)                                          \
      init_tables();                                                    \
    char str_id[INT_LENGTH * 4 + 4 + 1];                                \
    sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);                     \
    int previous = int_ht_get(tab_choice_instances, str_id);            \
    int instance;                                                       \
    if (table_miss) {                                                   \
      instance = 0;                                                     \
    } else {                                                            \
      instance = previous + 1;                                          \
    }                                                                   \
    int_ht_set(tab_choice_instances, str_id, instance);                 \
    int i;                                                              \
    for (i = 0; i < env_size; i++) {                                    \
      char name[MAX_NAME_LENGTH];                                       \
      char* env_fmt = "int!choice!%d!%d!%d!%d!%d!env!%s";               \
      sprintf(name, env_fmt, bl, bc, el, ec, instance, env_ids[i]);     \
      int sv;                                                           \
      klee_make_symbolic(&sv, sizeof(sv), name);                        \
      klee_assume(sv == env_vals[i]);                                   \
    }                                                                   \
                                                                        \
    char name[MAX_NAME_LENGTH];                                         \
    char* angelic_fmt = "%s!choice!%d!%d!%d!%d!%d!angelic";             \
    sprintf(name, angelic_fmt, typestr, bl, bc, el, ec, instance);      \
    int s;                                                              \
    klee_make_symbolic(&s, sizeof(s), name);                            \
                                                                        \
    return s;                                                           \
  }

CHOOSE_WITH_GUIDE_PROTO(bool, "bool")

#undef CHOOSE_WITH_GUIDE_PROTO

#define EXPR_PROTO(type, env_val_type, typename)                        \
  type angelix_expr_##typename(char *dc, type expr,                     \
                               int bl, int bc, int el, int ec,          \
                               char** env_ids,                          \
                               env_val_type* env_vals, int env_size) {  \
    if (getenv("ANGELIX_SYMBOLIC_RUNTIME")) {                           \
      switch(*dc) {                                                     \
      case 'A':                                                         \
      case 'P':                                                         \
        if (is_suspicious(bl, bc, el, ec)) {                            \
          return angelix_choose_##typename(bl, bc, el, ec,              \
                                           env_ids, env_vals,           \
                                           env_size);                   \
        }                                                               \
        else {                                                          \
          return angelix_trace_and_load_##typename(dc, expr,            \
                                                   bl, bc,              \
                                                   el, ec,              \
                                                   env_ids,             \
                                                   env_vals,            \
                                                   env_size);           \
        }                                                               \
      case 'I':                                                         \
      case 'L':                                                         \
      case 'G':                                                         \
      default:                                                          \
        if (getenv("ANGELIX_LOAD_JSON")) {                              \
          return angelix_trace_and_load_##typename(dc, expr,            \
                                                   bl, bc,              \
                                                   el, ec,              \
                                                   env_ids,             \
                                                   env_vals,            \
                                                   env_size);           \
        } else {                                                        \
          return angelix_choose_##typename(bl, bc, el, ec,              \
                                           env_ids, env_vals,           \
                                           env_size);                   \
        }                                                               \
      }                                                                 \
    }                                                                   \
    else {                                                              \
      return angelix_trace_and_load_##typename(dc, expr,                \
                                               bl, bc, el, ec,          \
                                               env_ids,                 \
                                               env_vals,                \
                                               env_size);               \
    }                                                                   \
  }

EXPR_PROTO(int, int, int)
EXPR_PROTO(bool, int, bool)
EXPR_PROTO(void*, void*, void_ptr)

#undef EXPR_PROTO


int angelix_ignore() {
  return 0;
}

#define TRACE_PROTO(type, env_val_type, typename)                       \
  void angelix_trace_##typename(char *trace_file, char* defect_class,   \
                                type expr,                              \
                                int bl, int bc, int el, int ec,         \
                                char** env_ids,                         \
                                env_val_type* env_vals, int env_size) { \
  int i = 0;                                                            \
  FILE *fp = fopen(trace_file, "a");                                    \
  if (fp == NULL) {                                                     \
    fprintf(stderr,                                                     \
            "[angelix runtime] angelix_trace file doesn't exist: %s\n", \
            trace_file);                                                \
    abort();                                                            \
  }                                                                     \
                                                                        \
  fprintf(fp, "%s, %d-%d-%d-%d, %d",                                    \
          defect_class, bl, bc, el, ec, expr);                          \
  fflush(fp);                                                           \
                                                                        \
  if (env_size > 0) {                                                   \
    fprintf(fp, ", ");                                                  \
    fflush(fp);                                                         \
  }                                                                     \
                                                                        \
  for (i = 0; i < env_size; i++) {                                      \
    fprintf(fp, "%s = %d", env_ids[i], env_vals[i]);                    \
    if (i < env_size - 1) {                                             \
      fprintf(fp, "; ");                                                \
      fflush(fp);                                                       \
    }                                                                   \
  }                                                                     \
                                                                        \
  fprintf(fp, "\n");                                                    \
  fflush(fp);                                                           \
  fclose(fp);                                                           \
}

TRACE_PROTO(int, int, int)
TRACE_PROTO(bool, int, bool)
// TRACE_PROTO(void*, void*, void_ptr)

#undef TRACE_PROTO

void angelix_trace_void_ptr(char *trace_file, char* defect_class,
                            void* expr,
                            int bl, int bc, int el, int ec,
                            char** env_ids,
                            void** env_vals, int env_size) {
  int i = 0;
  FILE *fp = fopen(trace_file, "a");
  if (fp == NULL) {
    fprintf(stderr,
            "[angelix runtime] angelix_trace file doesn't exist: %s\n",
            trace_file);
    abort();
  }

  fprintf(fp, "%s, %d-%d-%d-%d, %d",
          defect_class, bl, bc, el, ec, expr);
  fflush(fp);

  if (env_size > 0) {
    fprintf(fp, ", ");
    fflush(fp);
  }

  for (i = 0; i < env_size; i++) {
    fprintf(fp, "%s = %d", env_ids[i], env_vals[i]);
    if (i < env_size - 1) {
      fprintf(fp, "; ");
      fflush(fp);
    }
  }

  fprintf(fp, ", %d\n", env_size);
  fflush(fp);
  fclose(fp);
}


/*
 * versatile function. it can either trace or load
 */
#define TRACE_AND_LOAD_PROTO(type, env_val_type, typename)              \
  type angelix_trace_and_load_##typename(char* defect_class,            \
                                         type expr,                     \
                                         int bl, int bc,                \
                                         int el, int ec,                \
                                         char** env_ids,                \
                                         env_val_type* env_vals,        \
                                         int env_size) {                \
    char *trace_dir = getenv("ANGELIX_TRACE");                          \
    if (trace_dir) {                                                    \
      angelix_trace_##typename(trace_dir, defect_class, expr,           \
                              bl, bc, el, ec,                           \
                              env_ids, env_vals, env_size);             \
      return expr;                                                      \
    }                                                                   \
    else if (getenv("ANGELIX_LOAD") || getenv("ANGELIX_LOAD_JSON")) {   \
      if (getenv("ANGELIX_LOAD_JSON")) {                                \
        /* Only for branches */                                         \
        return angelix_load_json_##typename(defect_class, expr,         \
                                            bl, bc, el, ec,             \
                                            env_ids, env_vals,          \
                                            env_size);                  \
      } else {                                                          \
        return angelix_load_##typename(expr, bl, bc, el, ec);           \
      }                                                                 \
    }                                                                   \
    return expr;                                                        \
  }

TRACE_AND_LOAD_PROTO(int, int, int)
TRACE_AND_LOAD_PROTO(bool, int, bool)
// TRACE_AND_LOAD_PROTO(void*, void*, void_ptr)

#undef TRACE_AND_LOAD_PROTO

void* angelix_trace_and_load_void_ptr(char* defect_class,
                                     void* expr,
                                     int bl, int bc,
                                     int el, int ec,
                                     char** env_ids,
                                     void** env_vals,
                                     int env_size) {
  char *trace_dir = getenv("ANGELIX_TRACE");
  if (trace_dir) {
    angelix_trace_void_ptr(trace_dir, defect_class, expr,
                           bl, bc, el, ec,
                           env_ids, env_vals, env_size);
    return expr;
  }
  else if (getenv("ANGELIX_LOAD") || getenv("ANGELIX_LOAD_JSON")) {
    if (getenv("ANGELIX_LOAD_JSON")) {
      /* Only for branches and ptr */
      return angelix_load_json_void_ptr(defect_class, expr,
                                        bl, bc, el, ec,
                                        env_ids, env_vals,
                                        env_size);
    } else {
      return angelix_load_void_ptr(expr, bl, bc, el, ec);
    }
  }
  return expr;
}


#define ANGELIX_TRACE_FOR_SEARCH_PROTO(type, env_val_type, typename)    \
  void angelix_trace_for_search_##typename(char *trace_dir, type expr,  \
                                           char *dc,                    \
                                           int bl, int bc,              \
                                           int el, int ec,              \
                                           char** env_ids,              \
                                           env_val_type* env_vals,      \
                                           int env_size) {              \
    FILE *fp = fopen(trace_dir, "a");                                   \
    if (fp == NULL)                                                     \
      abort();                                                          \
    fprintf(fp, "%s, %d-%d-%d-%d, %d", dc, bl, bc, el, ec, expr);       \
                                                                        \
    if (env_size > 0) {                                                 \
      fprintf(fp, ", ");                                                \
    }                                                                   \
                                                                        \
    int i;                                                              \
    for (i = 0; i < env_size; i++) {                                    \
      fprintf(fp, "%s = %d", env_ids[i], env_vals[i]);                  \
      if (i < env_size - 1) {                                           \
        fprintf(fp, "; ");                                              \
      }                                                                 \
    }                                                                   \
                                                                        \
    fprintf(fp, "\n");                                                  \
    fclose(fp);                                                         \
  }

ANGELIX_TRACE_FOR_SEARCH_PROTO(int, int, int)
ANGELIX_TRACE_FOR_SEARCH_PROTO(bool, int, bool)
// ANGELIX_TRACE_FOR_SEARCH_PROTO(void*, void*, void_ptr)

#undef ANGELIX_TRACE_FOR_SEARCH_PROTO

void angelix_trace_for_search_void_ptr(char *trace_dir, void* expr,
                                       char *dc,
                                       int bl, int bc,
                                       int el, int ec,
                                       char** env_ids,
                                       void** env_vals,
                                       int env_size) {
  FILE *fp = fopen(trace_dir, "a");
  if (fp == NULL)
    abort();
  fprintf(fp, "%s, %d-%d-%d-%d, %d", dc, bl, bc, el, ec, expr);

  if (env_size > 0) {
    fprintf(fp, ", ");
  }

  int i;
  for (i = 0; i < env_size; i++) {
    fprintf(fp, "%s = %d", env_ids[i], env_vals[i]);
    if (i < env_size - 1) {
      fprintf(fp, "; ");
    }
  }

  fprintf(fp, ", %d\n", env_size);
  fclose(fp);
}


#define ANGELIX_LOAD_PROTO(type, typename)                              \
  type angelix_load_##typename(type expr,                               \
                               int bl, int bc, int el, int ec) {        \
    if (!tab_choice_instances)                                          \
      init_tables();                                                    \
    char str_id[INT_LENGTH * 4 + 4 + 1];                                \
    sprintf(str_id, "%d-%d-%d-%d", bl, bc, el, ec);                     \
    int previous = int_ht_get(tab_choice_instances, str_id);            \
    int instance;                                                       \
    if (table_miss) {                                                   \
      instance = 0;                                                     \
    } else {                                                            \
      instance = previous + 1;                                          \
    }                                                                   \
    int_ht_set(tab_choice_instances, str_id, instance);                 \
    char* data = load_instance(str_id, instance);                       \
    if (!data) {                                                        \
      return expr;                                                      \
    }                                                                   \
    type result = parse_int(data);                                      \
    return result;                                                      \
  }

ANGELIX_LOAD_PROTO(int, int)
ANGELIX_LOAD_PROTO(bool, bool)
ANGELIX_LOAD_PROTO(void*, void_ptr)

#undef ANGELIX_LOAD_PROTO


#define ANGELIX_LOAD_JSON_PROTO(type, env_val_type, typename)           \
  type angelix_load_json_##typename(char *dc, type expr,                \
                                    int bl, int bc, int el, int ec,     \
                                    char** env_ids,                     \
                                    env_val_type* env_vals,             \
                                    int env_size) {                     \
    if (tab_choice_instances == NULL) {                                 \
      init_tables();                                                    \
    }                                                                   \
    if (set_ids == NULL) {                                              \
      make_load_json_ready();                                           \
      init_set_ids();                                                   \
      srand(time(NULL));                                                \
    }                                                                   \
    char loc[INT_LENGTH * 4 + 4 + 1];                                   \
    sprintf(loc, "%d-%d-%d-%d", bl, bc, el, ec);                        \
                                                                        \
    assert(set_ids != NULL);                                            \
    if (!int_ht_has(set_ids, loc)) {                                    \
      /* return expr if not interested */                               \
      return expr;                                                      \
    }                                                                   \
                                                                        \
    int previous = int_ht_get(tab_choice_instances, loc);               \
    int instance;                                                       \
    if (table_miss) {                                                   \
      instance = 0;                                                     \
    } else {                                                            \
      instance = previous + 1;                                          \
    }                                                                   \
    int_ht_set(tab_choice_instances, loc, instance);                    \
    type result = load_instance_json(loc, instance);                    \
                                                                        \
    char *trace_dir = getenv("ANGELIX_TRACE_AFTER_LOAD");               \
    if (trace_dir) {                                                    \
      angelix_trace_for_search_##typename(trace_dir, result,            \
                                          dc, bl, bc, el, ec,           \
                                          env_ids, env_vals, env_size); \
    }                                                                   \
                                                                        \
    return result;                                                      \
  }

ANGELIX_LOAD_JSON_PROTO(int, int, int)
ANGELIX_LOAD_JSON_PROTO(bool, int, bool)
// ANGELIX_LOAD_JSON_PROTO(void*, void*, void_ptr)

#undef ANGELIX_LOAD_JSON_PROTO

void* angelix_load_json_void_ptr(char *dc, void* expr,
                                 int bl, int bc, int el, int ec,
                                 char** env_ids,
                                 void** env_vals,
                                 int env_size) {
  if (tab_choice_instances == NULL) {
    init_tables();
  }
  if (set_ids == NULL) {
    make_load_json_ready();
    init_set_ids();
    srand(time(NULL));
  }
  char loc[INT_LENGTH * 4 + 4 + 1];
  sprintf(loc, "%d-%d-%d-%d", bl, bc, el, ec);

  assert(set_ids != NULL);
  if (!int_ht_has(set_ids, loc)) {
    /* return expr if not interested */
    return expr;
  }

  int previous = int_ht_get(tab_choice_instances, loc);
  int instance;
  if (table_miss) {
    instance = 0;
  } else {
    instance = previous + 1;
  }

  // For pointers, we use a singleton list.
  // Hence, we assume that instance is always 0.
  instance = 0;
  int_ht_set(tab_choice_instances, loc, instance);
  int idx = load_instance_json(loc, instance);
  void* result = env_vals[idx];

  char *trace_dir = getenv("ANGELIX_TRACE_AFTER_LOAD");
  if (trace_dir) {
    angelix_trace_for_search_void_ptr(trace_dir, result,
                                      dc, bl, bc, el, ec,
                                      env_ids, env_vals, env_size);
  }

  return result;
}


bool is_suspicious(int bl, int bc, int el, int ec) {
  char loc[INT_LENGTH * 4 + 4 + 1];
  sprintf(loc, "%d-%d-%d-%d", bl, bc, el, ec);
  char *suspicious_rhses = getenv("ANGELIX_SUSPICIOUS_RHSES");
  if (suspicious_rhses != NULL) {
    char *rhs = strtok(suspicious_rhses, " ");
    while( rhs != NULL ) {
      if (strcmp(rhs, loc) == 0) {
        return true;
      }
      rhs = strtok(NULL, " ");
    }
    return false;
  } else {
    return false;
  }
}
