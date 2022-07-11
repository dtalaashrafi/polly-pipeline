/*
    * This file contains pipeline scheduling.

*/

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_space_private.h>
#include <isl_aff_private.h>
#include <isl/hash.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/schedule.h>
#include <isl_schedule_constraints.h>
#include <isl/schedule_node.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl/set.h>
#include <isl_union_set_private.h>
#include <isl_seq.h>
#include <isl_tab.h>
#include <isl_dim_map.h>
#include <isl/map_to_basic_set.h>
#include <isl_sort.h>
#include <isl_options_private.h>
#include <isl_tarjan.h>
#include <isl_morph.h>
#include <isl/ilp.h>
#include <isl_val_private.h>

// temp
#include <stdio.h>

isl_map * get_pipeline_relation(isl_map *rmap, isl_map *wmap)
{

   isl_map *map_read = isl_map_copy(rmap);
   isl_map *map_write = isl_map_copy(wmap);

   isl_map *write_reverse = isl_map_reverse(map_write);
   isl_map *K = isl_map_apply_range(map_read, write_reverse);
   isl_map *K_copy = isl_map_copy(K);
   
   isl_map *K_temp = isl_map_lexmax(K_copy);
   isl_set *D = isl_map_domain(K);
   isl_set *D_copy = isl_set_copy(D);

   isl_map *D1 = isl_set_lex_ge_set(D,D_copy);
   isl_map *L1 = isl_map_apply_range(D1,K_temp);
   isl_map *L = isl_map_lexmax(L1);

   // T = L.reverse().lexmax()
   isl_map *L_reverse = isl_map_reverse(L);
   isl_map *T = isl_map_lexmax(L_reverse);


   return T;
}

// blocks the set, w.r.t. the domain/range of the map
// choose = 0 => blocks the domain
// choose = 1 => blocks the range
isl_map * get_blocks(isl_set *s, isl_map *m, int choose)
{
   isl_set *s_copy = isl_set_copy(s);
   isl_map *m_copy = isl_map_copy(m);

   isl_set *D;
   if(choose == 0)
      D = isl_map_domain(m_copy);
   else if(choose == 1)
      D = isl_map_range(m_copy);
      
   isl_map *Dp1 = isl_set_lex_ge_set(D,s_copy);
   isl_map *Dp = isl_map_reverse(Dp1);
   isl_map *E = isl_map_lexmin(Dp);

   return E;
}


isl_schedule * get_pipeline_schedule_per_loop_nest(isl_map *E)
{
   isl_map *E_copy = isl_map_copy(E);
   isl_map *E_copy_1 = isl_map_copy(E);
   isl_map *E_copy_2 = isl_map_copy(E);
   isl_map *EE = isl_map_coalesce(E);

   isl_ctx *ctx = isl_map_get_ctx(E);
   isl_id *new_id = isl_id_read_from_str(ctx,"task");
   // isl_id * new_id = make_id_with_map_as_user(ctx, EE);


   isl_set *rE = isl_map_range(E_copy);
   isl_set *dE = isl_map_domain(E_copy_1);

   isl_set *rE_copy = isl_set_copy(rE);
   isl_union_set *rE_uset = isl_union_set_from_set(rE_copy);

   isl_set *dE_copy = isl_set_copy(dE);
   isl_union_set *dE_uset = isl_union_set_from_set(dE_copy);

   isl_map *rE_map = isl_set_flatten_map(rE);
   isl_map *dE_map = isl_set_flatten_map(dE);
   

   isl_union_map *E_umap = isl_union_map_from_map(E_copy_2);
   isl_union_map *E_umap_copy = isl_union_map_copy(E_umap);
   isl_union_map *rE_umap = isl_union_map_from_map(rE_map);
   isl_union_map *dE_umap = isl_union_map_from_map(dE_map);

   isl_multi_union_pw_aff *ps = isl_multi_union_pw_aff_from_union_map(E_umap);
   isl_multi_union_pw_aff *ps1 = isl_multi_union_pw_aff_from_union_map(dE_umap);
   isl_multi_union_pw_aff *ps2 = isl_multi_union_pw_aff_from_union_map(rE_umap);

   isl_schedule_node *r_sch_node = isl_schedule_node_from_domain(rE_uset);
   isl_schedule_node *r_sch_node_child = isl_schedule_node_child(r_sch_node,0);

   isl_schedule_node *r_h_node = isl_schedule_node_insert_partial_schedule(r_sch_node_child, ps2);
   isl_schedule_node *r_h_node_1 = isl_schedule_node_band_member_set_ast_loop_type(r_h_node,1,1); 
   isl_schedule_node *r_h_node_2 = isl_schedule_node_band_member_set_ast_loop_type(r_h_node_1,0,1); 
   isl_schedule *r_h = isl_schedule_node_get_schedule(r_h_node_2);

   // isl_schedule_node *r_h_node = isl_schedule_node_insert_partial_schedule(r_sch_node_child, ps2);
   // isl_schedule *r_h = isl_schedule_node_get_schedule(r_h_node);

   isl_schedule_node *exp_sch_node_2 = isl_schedule_node_from_domain(dE_uset);
   isl_schedule_node *exp_sch_node_child = isl_schedule_node_child(exp_sch_node_2,0);
   isl_schedule_node *exp_sch_node_1 = isl_schedule_node_insert_partial_schedule(exp_sch_node_child,ps1);
   isl_schedule_node *exp_sch_node = isl_schedule_node_insert_mark(exp_sch_node_1,new_id);
   isl_schedule *exp_sch = isl_schedule_node_get_schedule(exp_sch_node);

   isl_union_pw_multi_aff *contraction = isl_union_pw_multi_aff_from_union_map(E_umap_copy);

   // final_sch = rt_h_sch.expand(cont, exp_sch.get_schedule())
   isl_schedule *final_sch = isl_schedule_expand(r_h,contraction,exp_sch);

   return final_sch;

}





isl_union_map  *isl_schedule_pipeline(isl_ctx *ctx, isl_map *read_map, isl_map *write_map, isl_set *read_dom, isl_set *write_dom)
{

   printf("******************************************************************\n");

   // isl_ctx *ctx = isl_ctx_alloc();


   // isl_set_list *dom_list = isl_union_set_get_set_list(dom);
   // isl_union_map_list_dump(dom_list);
   // how to know which is which?
   isl_set *write_copy = isl_set_copy(write_dom);
   // isl_map *wr_dom = isl_map_from_domain(write_map);
   // isl_map *rd_dom = isl_set_flatten_map(read_map);

   // isl_map *wmap = isl_map_apply_domain(write_map, wr_dom);
   // isl_map *rmap = isl_map_apply_domain(read_map, rd_dom);


   // isl_map *wmap_copy = isl_map_copy(wmap);
   // isl_map *rmap_copy = isl_map_copy(rmap);
   // // isl_union_map *dep_copy = isl_map_copy(dep);

   // isl_map *wmap_copy_1 = isl_map_copy(wmap);
   // isl_map *rmap_copy_1 = isl_map_copy(rmap);  
   // // isl_union_map *dep_copy_1 = isl_map_copy(dep);

   // isl_set *I = isl_map_domain(wmap_copy);
   // isl_set *J = isl_map_domain(rmap_copy);

   // isl_map *T = get_pipeline_relation(rmap_copy_1, wmap_copy_1);
   // isl_map *T_copy = isl_map_copy(T);
   // isl_map *T_copy_1 = isl_map_copy(T);

   // isl_map *E = get_blocks(I,T_copy, 0);
   // isl_map *F = get_blocks(J,T,1);

   // isl_schedule *E_task_sch = get_pipeline_schedule_per_loop_nest(E);
   // isl_schedule *F_task_sch = get_pipeline_schedule_per_loop_nest(F);

   // // printf("schedules from the isl function:\n*************\n");
   // // isl_schedule_dump(final_sch);
   // // printf("**************************\n");
   // // isl_schedule_dump(F_task_sch);

   // //get map of the two sch, and find its union
   // isl_union_map *E_map = isl_schedule_get_map(E_task_sch);
   // isl_union_map *F_map = isl_schedule_get_map(F_task_sch);

   // isl_union_map *final_sch_map = isl_union_map_union(E_map, F_map);


   // return final_sch_map;

}


// isl_schedule  *isl_schedule_pipeline(isl_map *rmap, isl_map *wmap_str, isl_union_set *dom)
// {
//    isl_ctx *ctx = isl_ctx_alloc();

//    // char *domain_str = "{ Stmt1[i0, i1] : 0 <= i0 <= 19 and 0 <= i1 <= 19 }";
//    // isl_set *dom = isl_set_read_from_str(ctx,domain_str);

//    // isl_schedule_node *sch_node = isl_schedule_node_from_domain(isl_union_set_from_set(dom));

//    // isl_multi_union_pw_aff *ps = isl_multi_union_pw_aff_from_union_map(isl_union_map_from_map(isl_map_read_from_str(ctx,"{Stmt1[i0,i1]->[i0,i1]}")));
//    // sch_node = isl_schedule_node_insert_partial_schedule(isl_schedule_node_get_child(sch_node,0) , ps);

//    // isl_schedule *final_schedule = isl_schedule_node_get_schedule(sch_node);

//    // return final_schedule;

// }